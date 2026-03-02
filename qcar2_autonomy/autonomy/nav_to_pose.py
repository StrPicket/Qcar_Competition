#! /usr/bin/env python3

import time
import math
import numpy as np
import scipy.signal as signal
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState, Image
from std_msgs.msg import Bool, Float32
from rcl_interfaces.msg import SetParametersResult
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import cv2
from cv_bridge import CvBridge

'''
Waypoint follower para QCar 2 — carril DERECHO con detección HSV.

Lógica de carril:
  - El robot circula por el CARRIL DERECHO.
  - La línea AMARILLA (central) queda a la IZQUIERDA del robot.
  - La línea BLANCA (orilla) queda a la DERECHA del robot.

  Caso A — ambas líneas visibles:
      El robot apunta al centro del espacio entre ellas.

  Caso B — solo línea amarilla visible:
      El robot se mantiene a SINGLE_LINE_OFFSET de su lado derecho.

  Caso C — solo línea blanca visible:
      El robot se mantiene a SINGLE_LINE_OFFSET de su lado izquierdo.

  Caso D — ninguna línea visible:
      Sin corrección de carril; solo waypoints.

La corrección lateral se fusiona con el steering del waypoint follower
mediante el parámetro LANE_WEIGHT.  Si la detección falla (Case D),
LANE_WEIGHT se fuerza a 0 automáticamente.
'''

# ============================================================
# WAYPOINTS [x, y] en metros (frame map_rotated)
# ============================================================
WAYPOINTS = [
    [0,    -0.85],
    [2.35, -0.35],
    [2.82,  0.00],
    [2.925,     5.50],
    [2.87,   6.1],
    [2.4,   6.48],
    [-1.1,  7.22],
    [-2.2,  6.9],
    [-3.1,  4.50],
    [-3.6,  3.35],
    [-1,    1.4],
]

# ============================================================
# GANANCIAS — Waypoint follower
# ============================================================
Kp_vel  = 0.1
Ki_vel  = 0.2
V_MAX   = 0.4       # [m/s]

Kp_theta = 20.4
Kv_theta = 2.3

LOOKAHEAD_BASE = 0.12   # [m]
LOOKAHEAD_K    = 0.40   # [s]
BLEND_DIST     = 0.3    # [m]
BRAKE_DIST     = 0.8    # [m]
V_APPROACH     = 0.15   # [m/s]
STOP_THRESHOLD = 0.05   # [m]
MAX_STEERING   = math.pi / 4  # [rad]

# ============================================================
# PARÁMETROS — Detección de carril HSV
# ============================================================

CAMERA_TOPIC = '/qcar2_camera/image_raw'

# ── ROI ──────────────────────────────────────────────────────
# Analiza solo el 35 % inferior de la imagen.
# Aumenta si las líneas aparecen muy arriba en el frame.
# Disminuye si incluye demasiado ruido del entorno.
ROI_FRACTION = 0.35

# ── HSV — Línea AMARILLA ─────────────────────────────────────
# Amarillo puro en HSV: H≈25-35, S>100, V>100.
# Si la iluminación es cálida/fría ajusta H ±5.
YELLOW_HSV_LOW  = np.array([ 15,  80,  80], dtype=np.uint8)
YELLOW_HSV_HIGH = np.array([ 40, 255, 255], dtype=np.uint8)

# ── HSV — Línea BLANCA ───────────────────────────────────────
# Blanco: S muy bajo, V muy alto.
# Si hay reflejos o la pista es muy brillante, sube LOW[2] a 200.
WHITE_HSV_LOW  = np.array([  0,   0, 180], dtype=np.uint8)
WHITE_HSV_HIGH = np.array([179,  55, 255], dtype=np.uint8)

# ── Área mínima válida [px²] ──────────────────────────────────
LINE_MIN_AREA = 300

# ── Ganancia de corrección lateral ───────────────────────────
# Convierte error normalizado [-1,1] a ángulo de rueda [rad].
# Sube si el robot no corrige suficiente; baja si zigzaguea.
Kp_lane = 0.5

# ── Peso del lane en la fusión con waypoints ──────────────────
# 0.0 → solo waypoints | 1.0 → solo lane
# 0.35 es un buen punto de partida: waypoints guían la ruta
# global y el lane ajusta la posición lateral dentro del carril.
LANE_WEIGHT = 0.35

# ── Offset de referencia con UNA sola línea visible ──────────
# Fracción del ancho de imagen que el robot mantiene de distancia
# a la única línea detectada.  0.25 ≈ ¼ del ancho de imagen.
SINGLE_LINE_OFFSET = 0.25

# ── Imagen de debug ───────────────────────────────────────────
PUBLISH_DEBUG = True


class PathFollower(Node):

    def __init__(self):
        super().__init__('path_follower')

        # ── Parámetros ROS ──────────────────────────────────────────
        self.declare_parameter('desired_speed', [0.4])
        self.desired_speed = list(
            self.get_parameter('desired_speed').get_parameter_value().double_array_value
        )

        self.declare_parameter('start_path', [True])
        self.path_execute_flag = list(
            self.get_parameter('start_path').get_parameter_value().bool_array_value
        )[0]

        self.add_on_set_parameters_callback(self.parameter_update_callback)

        # ── TF2 ─────────────────────────────────────────────────────
        self.target_frame = self.declare_parameter(
            'target_frame', 'base_link'
        ).get_parameter_value().string_value

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Pose ─────────────────────────────────────────────────────
        self.pos_x        = -1.25
        self.pos_y        = -0.83
        self.yaw          = -25 * math.pi / 180.0
        self.tf_available = False

        self.x_kin  = -1.25
        self.y_kin  = -0.83
        self.th_kin = -25 * math.pi / 180.0

        # ── Filtro giroscopio ────────────────────────────────────────
        self.dt = 1 / 80
        self._init_gyro_filter(cutoff_hz=15.0)

        # ── Waypoints ────────────────────────────────────────────────
        self.wp  = np.array(WAYPOINTS, dtype=float)
        self.N   = len(self.wp)
        self.wpi = 0

        # ── Controlador de velocidad ──────────────────────────────────
        self.integral_vel = 0.0
        self.t_prev_vel   = time.time()

        # ── Steering ─────────────────────────────────────────────────
        self.current_steering = 0.0

        # ── Lane detection ────────────────────────────────────────────
        self.bridge          = CvBridge()
        self.lane_correction = 0.0    # [rad]
        self.lane_valid      = False
        self.lane_mode       = 'NONE' # BOTH | YELLOW | WHITE | NONE

        # ── Flags / sensores ──────────────────────────────────────────
        self.motion_flag    = True
        self.path_complete  = False
        self.speed_mult     = 1.0
        self.gyroscope      = [0.0, 0.0, 0.0]
        self.measured_speed = 0.0

        # ── Publishers ────────────────────────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist, '/cmd_vel_nav',  1)
        self.status_pub = self.create_publisher(Bool,  '/path_status',  1)
        if PUBLISH_DEBUG:
            self.debug_pub = self.create_publisher(
                Image, '/lane_debug/image_raw', 1
            )

        # ── Subscribers ───────────────────────────────────────────────
        self.create_subscription(JointState, '/qcar2_joint',   self._joint_cb,  1)
        self.create_subscription(Imu,        '/qcar2_imu',     self._imu_cb,   10)
        self.create_subscription(Bool,       '/motion_enable', self._motion_cb, 1)
        self.create_subscription(Float32,    '/speed_limit',   self._speed_cb,  1)
        self.create_subscription(Image,       CAMERA_TOPIC,    self._camera_cb, 1)

        # ── Timers ────────────────────────────────────────────────────
        self.create_timer(self.dt,     self._tf_timer)
        self.create_timer(self.dt,     self._kinematic_integrator)
        self.create_timer(self.dt,     self._control_loop)
        self.create_timer(1.0 / 30.0, self._status_log)

        self.get_logger().info(
            f'PathFollower — carril derecho + HSV | '
            f'{self.N} wps | v={self.desired_speed[0]} m/s | '
            f'lane_weight={LANE_WEIGHT}'
        )

    # ════════════════════════════════════════════════════════════════
    # DETECCIÓN DE CARRIL
    # ════════════════════════════════════════════════════════════════

    def _centroid_x(self, mask):
        """Coordenada X del centroide de una máscara, o None si no hay área."""
        if int(np.sum(mask > 0)) < LINE_MIN_AREA:
            return None
        M = cv2.moments(mask)
        return (M['m10'] / M['m00']) if M['m00'] > 0 else None

    def _camera_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge: {e}')
            return

        h, w = frame.shape[:2]

        # ── ROI inferior ─────────────────────────────────────────────
        roi_y = int(h * (1.0 - ROI_FRACTION))
        roi   = frame[roi_y:h, :]
        rh, rw = roi.shape[:2]

        # ── Máscaras HSV ─────────────────────────────────────────────
        hsv    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        def clean(mask):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            return mask

        mask_yellow = clean(cv2.inRange(hsv, YELLOW_HSV_LOW,  YELLOW_HSV_HIGH))
        mask_white  = clean(cv2.inRange(hsv, WHITE_HSV_LOW,   WHITE_HSV_HIGH))

        # ── Centroide de cada línea ───────────────────────────────────
        x_yellow = self._centroid_x(mask_yellow)  # debe quedar a la IZQUIERDA
        x_white  = self._centroid_x(mask_white)   # debe quedar a la DERECHA

        center = rw / 2.0

        # ── Error lateral y modo ──────────────────────────────────────
        #   error > 0  → robot demasiado a la derecha → corrección izquierda
        #   error < 0  → robot demasiado a la izquierda → corrección derecha

        if x_yellow is not None and x_white is not None:
            if x_white > x_yellow:
                # Caso A: ambas líneas en la posición esperada
                # (carril derecho: blanca a la derecha, amarilla a la izquierda)
                lane_center      = (x_yellow + x_white) / 2.0
                error            = (center - lane_center) / (rw / 2.0)
                self.lane_mode   = 'BOTH'
                self.lane_valid  = True
            else:
                # Detección cruzada (iluminación anómala) → ignorar
                error            = 0.0
                self.lane_mode   = 'INVERT'
                self.lane_valid  = False

        elif x_yellow is not None:
            # Caso B: solo amarilla → mantenerse a su derecha
            ref              = x_yellow + SINGLE_LINE_OFFSET * rw
            error            = (center - ref) / (rw / 2.0)
            self.lane_mode   = 'YELLOW'
            self.lane_valid  = True

        elif x_white is not None:
            # Caso C: solo blanca → mantenerse a su izquierda
            ref              = x_white - SINGLE_LINE_OFFSET * rw
            error            = (center - ref) / (rw / 2.0)
            self.lane_mode   = 'WHITE'
            self.lane_valid  = True

        else:
            # Caso D: sin detección
            self.lane_correction = 0.0
            self.lane_valid      = False
            self.lane_mode       = 'NONE'
            self._publish_debug(roi, mask_yellow, mask_white,
                                x_yellow, x_white, None, rw, rh)
            return

        # corrección: error > 0 → giro izquierda → steering negativo en ROS
        self.lane_correction = float(
            np.clip(-Kp_lane * error, -MAX_STEERING, MAX_STEERING)
        )

        self._publish_debug(roi, mask_yellow, mask_white,
                            x_yellow, x_white, error, rw, rh)

    # ── Imagen de debug ───────────────────────────────────────────

    def _publish_debug(self, roi, mask_y, mask_w,
                       x_yellow, x_white, error, rw, rh):
        if not PUBLISH_DEBUG:
            return

        debug = roi.copy()

        # Superposición coloreada de máscaras
        ov_y = debug.copy()
        ov_y[mask_y > 0] = (0, 215, 255)       # BGR ≈ amarillo
        cv2.addWeighted(ov_y, 0.45, debug, 0.55, 0, debug)

        ov_w = debug.copy()
        ov_w[mask_w > 0] = (230, 230, 230)      # BGR ≈ blanco
        cv2.addWeighted(ov_w, 0.45, debug, 0.55, 0, debug)

        # Centro geométrico de la imagen (verde)
        cv2.line(debug, (rw // 2, 0), (rw // 2, rh), (0, 200, 0), 1)

        # Línea amarilla detectada
        if x_yellow is not None:
            xi = int(x_yellow)
            cv2.line(debug, (xi, 0), (xi, rh), (0, 200, 255), 2)
            cv2.putText(debug, 'Y', (xi + 4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Línea blanca detectada
        if x_white is not None:
            xi = int(x_white)
            cv2.line(debug, (xi, 0), (xi, rh), (255, 255, 255), 2)
            cv2.putText(debug, 'W', (xi + 4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Centro del carril (naranja) cuando ambas líneas son válidas
        if (x_yellow is not None and x_white is not None
                and x_yellow > x_white):
            lc = int((x_yellow + x_white) / 2)
            cv2.line(debug, (lc, 0), (lc, rh), (0, 100, 255), 2)

        # HUD inferior
        color_hud = (0, 255, 0) if self.lane_valid else (0, 0, 255)
        cv2.putText(
            debug,
            f'{self.lane_mode}  corr={math.degrees(self.lane_correction):.1f}deg',
            (6, rh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color_hud, 2,
        )
        if error is not None:
            cv2.putText(
                debug, f'err={error:.3f}',
                (6, rh - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1,
            )

        try:
            self.debug_pub.publish(
                self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
            )
        except Exception:
            pass

    # ════════════════════════════════════════════════════════════════
    # LOG DE ESTADO
    # ════════════════════════════════════════════════════════════════

    def _status_log(self):
        px, py, th = (self.pos_x, self.pos_y, self.yaw) if self.tf_available \
                    else (self.x_kin, self.y_kin, self.th_kin)
        fuente = 'TF2' if self.tf_available else 'KIN'

        wp_txt, dist_txt = 'COMPLETO', '-'
        if not self.path_complete and self.wpi < self.N:
            wp       = self.wp[self.wpi]
            dist     = math.hypot(wp[0] - px, wp[1] - py)
            wp_txt   = f'{self.wpi + 1}/{self.N} @ ({wp[0]:.2f},{wp[1]:.2f})'
            dist_txt = f'{dist:.3f} m'

        lane_txt = (
            f'{self.lane_mode} {math.degrees(self.lane_correction):.1f}°'
            if self.lane_valid else 'NONE'
        )

        self.get_logger().info(
            f'[STATUS] pose=({px:.3f},{py:.3f}) yaw={math.degrees(th):.1f}° [{fuente}] | '
            f'v={self.measured_speed:.3f}m/s | '
            f'steer={math.degrees(self.current_steering):.1f}° | '
            f'lane={lane_txt} | '
            f'wp={wp_txt} dist={dist_txt} | '
            f"motion={'ON' if self.motion_flag else 'OFF'}"
        )

    # ════════════════════════════════════════════════════════════════
    # PARÁMETROS DINÁMICOS
    # ════════════════════════════════════════════════════════════════

    def parameter_update_callback(self, params):
        for param in params:
            if param.name == 'desired_speed' and param.type_ == param.Type.DOUBLE_ARRAY:
                self.desired_speed = list(param.value)
            elif param.name == 'start_path' and param.type_ == param.Type.BOOL_ARRAY:
                self.path_execute_flag = list(param.value)[0]
                if self.path_execute_flag:
                    self.wpi           = 0
                    self.path_complete = False
                    self.integral_vel  = 0.0
                    self.get_logger().info('Path reiniciado.')
                else:
                    self.get_logger().info('Path detenido.')
        return SetParametersResult(successful=True)

    # ════════════════════════════════════════════════════════════════
    # CALLBACKS SENSORES
    # ════════════════════════════════════════════════════════════════

    def _joint_cb(self, msg):
        self.measured_speed = (
            msg.velocity[0] / (720.0 * 4.0)
        ) * ((13.0 * 19.0) / (70.0 * 30.0)) * (2.0 * math.pi) * 0.033

    def _imu_cb(self, msg):
        self.gyroscope = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
        ]

    def _motion_cb(self, msg):
        self.motion_flag = msg.data

    def _speed_cb(self, msg):
        self.speed_mult = msg.data

    # ════════════════════════════════════════════════════════════════
    # TF2
    # ════════════════════════════════════════════════════════════════

    def _tf_timer(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map_rotated', self.target_frame, rclpy.time.Time()
            )
            self.pos_x = t.transform.translation.x
            self.pos_y = t.transform.translation.y
            quat = [
                t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w,
            ]
            _, _, self.yaw = R.from_quat(quat).as_euler('xyz')
            self.tf_available = True
            self.x_kin  = self.pos_x
            self.y_kin  = self.pos_y
            self.th_kin = self.yaw
        except TransformException:
            self.tf_available = False

    # ════════════════════════════════════════════════════════════════
    # INTEGRADOR CINEMÁTICO
    # ════════════════════════════════════════════════════════════════

    def _kinematic_integrator(self):
        if not self.tf_available:
            omega       = self.gyroscope[2]
            self.x_kin  += self.dt * self.measured_speed * math.cos(self.th_kin)
            self.y_kin  += self.dt * self.measured_speed * math.sin(self.th_kin)
            self.th_kin += self.dt * omega
            self.th_kin  = math.atan2(math.sin(self.th_kin), math.cos(self.th_kin))

    # ════════════════════════════════════════════════════════════════
    # FILTRO BUTTERWORTH
    # ════════════════════════════════════════════════════════════════

    def _init_gyro_filter(self, cutoff_hz):
        nyq             = 0.5 / self.dt
        b, a            = signal.butter(2, cutoff_hz / nyq)
        self._filt_b    = b
        self._filt_a    = a
        self._filt_hist = {'in': [0.0] * 3, 'out': [0.0] * 3}

    def _gyro_filter(self, raw):
        h = self._filt_hist
        h['in'] = [raw] + h['in'][:2]
        y = (
            self._filt_b[0] * h['in'][0]
            + self._filt_b[1] * h['in'][1]
            + self._filt_b[2] * h['in'][2]
            - self._filt_a[1] * h['out'][0]
            - self._filt_a[2] * h['out'][1]
        )
        h['out'] = [y] + h['out'][:2]
        return y

    # ════════════════════════════════════════════════════════════════
    # BUCLE DE CONTROL  (80 Hz)
    # ════════════════════════════════════════════════════════════════

    def _control_loop(self):
        enable        = 0.0
        speed_command = 0.0

        if self.path_execute_flag and self.motion_flag and not self.path_complete:
            enable = 1.0

            if self.tf_available:
                px, py, th = self.pos_x, self.pos_y, self.yaw
            else:
                px, py, th = self.x_kin, self.y_kin, self.th_kin

            # ── Lookahead dinámico ───────────────────────────────────
            is_last   = (self.wpi == self.N - 1)
            lookahead = (
                STOP_THRESHOLD if is_last
                else LOOKAHEAD_BASE + abs(self.measured_speed) * LOOKAHEAD_K
            )

            wp   = self.wp[self.wpi]
            dx   = wp[0] - px
            dy   = wp[1] - py
            dist = math.hypot(dx, dy)

            # ── Avanzar waypoint ─────────────────────────────────────
            if dist < lookahead:
                if not is_last:
                    self.get_logger().info(
                        f'WP {self.wpi + 1}/{self.N} capturado (dist={dist:.3f} m)'
                    )
                    self.wpi += 1
                    wp      = self.wp[self.wpi]
                    dx      = wp[0] - px
                    dy      = wp[1] - py
                    dist    = math.hypot(dx, dy)
                    is_last = (self.wpi == self.N - 1)
                else:
                    self.path_complete    = True
                    self.current_steering = 0.0
                    self.integral_vel     = 0.0
                    self.get_logger().info('Recorrido completo.')
                    self._publish(0.0, 0.0)
                    self._publish_status()
                    return

            # ── Ángulo deseado con blend ─────────────────────────────
            theta_curr = math.atan2(dy, dx)

            if not is_last and self.wpi + 1 < self.N and dist < BLEND_DIST:
                alpha     = 1.0 - dist / BLEND_DIST
                wp_next   = self.wp[self.wpi + 1]
                theta_nxt = math.atan2(wp_next[1] - py, wp_next[0] - px)
                dth       = math.atan2(
                    math.sin(theta_nxt - theta_curr),
                    math.cos(theta_nxt - theta_curr)
                )
                theta_des = theta_curr + alpha * dth
            else:
                theta_des = theta_curr

            # ── Steering waypoint (PV) ────────────────────────────────
            omega_raw   = self._gyro_filter(self.gyroscope[2])
            omega       = omega_raw * math.pi / 180.0
            error_theta = math.atan2(
                math.sin(theta_des - th),
                math.cos(theta_des - th)
            )
            steering_wp = Kp_theta * error_theta - Kv_theta * omega

            # ── Fusión waypoint + corrección de carril ─────────────────
            #
            #   steering_final = (1 - w) * steering_wp  +  w * lane_correction
            #
            #   Si no hay detección válida → w = 0 → solo waypoints.
            #   Si hay detección → w = LANE_WEIGHT (ajustable arriba).
            #
            w = LANE_WEIGHT if self.lane_valid else 0.0

            self.current_steering = float(np.clip(
                (1.0 - w) * steering_wp + w * self.lane_correction,
                -MAX_STEERING, MAX_STEERING
            ))

            # ── Velocidad ─────────────────────────────────────────────
            if is_last and dist < BRAKE_DIST:
                v_ref_base = (
                    V_APPROACH
                    + (self.desired_speed[0] - V_APPROACH) * (dist / BRAKE_DIST)
                )
            else:
                v_ref_base = self.desired_speed[0]

            v_ref = self.speed_mult * v_ref_base

            t_now           = time.time()
            dt_vel          = max(t_now - self.t_prev_vel, 1e-6)
            self.t_prev_vel = t_now

            v_real    = abs(self.measured_speed)
            error_vel = v_ref - v_real

            if self.speed_mult > 0.0:
                self.integral_vel += error_vel * dt_vel
                self.integral_vel  = np.clip(
                    self.integral_vel, -V_MAX / Ki_vel, V_MAX / Ki_vel
                )
            else:
                self.integral_vel = 0.0

            speed_command = float(np.clip(
                Ki_vel * self.integral_vel - Kp_vel * v_real, 0.0, V_MAX
            ))

        else:
            self.integral_vel     = 0.0
            self.current_steering = 0.0

        self._publish(enable, speed_command)
        self._publish_status()

    # ════════════════════════════════════════════════════════════════
    # PUBLICADORES
    # ════════════════════════════════════════════════════════════════

    def _publish(self, enable, speed_command):
        msg = Twist()
        if speed_command > 0.0:
            msg.linear.x = self.speed_mult * enable * float(np.clip(
                speed_command * math.cos(self.current_steering) ** 2,
                0.05, 0.7
            ))
        else:
            msg.linear.x = 0.0
        msg.angular.z = enable * self.current_steering
        self.cmd_pub.publish(msg)

    def _publish_status(self):
        msg      = Bool()
        msg.data = self.path_complete
        self.status_pub.publish(msg)


# ================================================================

def main():
    rclpy.init()
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()