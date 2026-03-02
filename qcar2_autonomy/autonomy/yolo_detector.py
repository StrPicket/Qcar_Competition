#! /usr/bin/env python3

# Fix para os.getlogin() fallando en contenedores Docker
import os
import getpass
os.getlogin = getpass.getuser

# Quanser specific packages
from pit.YOLO.nets import YOLOv8
from pit.YOLO.utils import QCar2DepthAligned

# Generic python packages
import time
import numpy as np
import cv2

# ROS specific packages
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

'''
Description:
Node for detecting traffic light state and signs on the road. Provides flags
which define if a traffic signal has been detected and what action to take.

[MERGED] - Lógica completa del v1 (semáforos + speed_limit) + bbox robusto del v2
'''

class ObjectDetector(Node):

    def __init__(self):
        super().__init__('yolo_detector')
        self.get_logger().info("="*60)
        self.get_logger().info("[INIT] Iniciando ObjectDetector (MERGED MODE)")
        self.get_logger().info("="*60)

        imageWidth  = 640
        imageHeight = 480

        # ── Cámara ──────────────────────────────────────────────────
        self.get_logger().info("[INIT] Abriendo QCar2DepthAligned...")
        try:
            self.QCarImg = QCar2DepthAligned()
            self.get_logger().info("[INIT] QCar2DepthAligned OK")
        except Exception as e:
            self.get_logger().error(f"[INIT] FALLO QCar2DepthAligned: {e}")
            raise

        # ── Modelo YOLO ─────────────────────────────────────────────
        self.get_logger().info(f"[INIT] Cargando YOLOv8  ({imageWidth}x{imageHeight})...")
        try:
            self.myYolo = YOLOv8(
                imageHeight=imageHeight,
                imageWidth=imageWidth,
            )
            self.get_logger().info("[INIT] YOLOv8 OK")
        except Exception as e:
            self.get_logger().error(f"[INIT] FALLO YOLOv8: {e}")
            raise

        # ── Publishers ──────────────────────────────────────────────
        self.motion_publisher   = self.create_publisher(Bool,    '/motion_enable',          1)
        self.publish_speed      = self.create_publisher(Float32, '/speed_limit',            10)
        self.publish_rgb        = self.create_publisher(Image,   '/qcar_camera/rgb',        10)
        self.publish_depth      = self.create_publisher(Image,   '/qcar_camera/depth',      10)
        self.publish_detections = self.create_publisher(Image,   '/qcar_camera/detections', 10)

        self.get_logger().info("[INIT] Publishers creados:")
        self.get_logger().info("        /motion_enable")
        self.get_logger().info("        /speed_limit")
        self.get_logger().info("        /qcar_camera/rgb")
        self.get_logger().info("        /qcar_camera/depth")
        self.get_logger().info("        /qcar_camera/detections")

        # ── Estado ──────────────────────────────────────────────────
        self.bridge             = CvBridge()
        self.motion_enable      = True
        self.detection_cooldown = 10.0
        self.disable_until      = 0.0
        self.flag_value         = False
        self.sign_detected      = False
        self.speed_limit        = 1.0
        self.t0                 = time.time()
        self._frame_count       = 0
        self._detect_count      = 0
        self._last_log_t        = time.time()

        # ── Debug controls ───────────────────────────────────────────
        self.debug_keys_once    = True   # imprime keys solo la primera vez que falte bbox

        self.publish_motion_flag(True)

        # ── Timers ──────────────────────────────────────────────────
        self.dt     = 1 / 30
        self.timer  = self.create_timer(self.dt,  self.on_timer)
        self.timer2 = self.create_timer(1 / 500,  self.flag_publisher)

        self.get_logger().info("[INIT] Timers creados. Nodo listo.\n")

    # ==========================================================
    # BBOX UTILITIES  (del v2)
    # ==========================================================
    def _to_pixel_bbox(self, x1, y1, x2, y2, img_w, img_h):
        """Convierte bbox a píxeles si viene normalizado y lo recorta a la imagen."""
        if (0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and
                0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0):
            x1 *= img_w; x2 *= img_w
            y1 *= img_h; y2 *= img_h

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        x1 = int(max(0, min(img_w - 1, round(x1))))
        x2 = int(max(0, min(img_w - 1, round(x2))))
        y1 = int(max(0, min(img_h - 1, round(y1))))
        y2 = int(max(0, min(img_h - 1, round(y2))))

        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    # Tamaño estimado de caja según tipo de objeto (en píxeles a 640x480)
    _BOX_SIZES = {
        "traffic light": (20, 30),   # ancho, alto
        "stop sign":     (50, 50),
        "yield sign":    (50, 50),
        "car":           (80, 50),
    }
    _BOX_DEFAULT = (40, 40)

    def _extract_bbox_xyxy(self, attrs, img_w, img_h):
        """
        Regresa bbox (x1,y1,x2,y2) en píxeles o None.

        El wrapper de Quanser entrega: name, distance, conf, x, y, lightColor
        donde x,y es el centro del objeto en píxeles.
        Si existen campos xyxy/bbox clásicos también los intenta primero.
        """
        # 1) Campos xyxy directos (por si cambia el wrapper en el futuro)
        for key in ["bbox", "xyxy", "tlbr", "rect"]:
            if key in attrs and attrs[key] is not None:
                b = attrs[key]
                try:
                    return self._to_pixel_bbox(
                        float(b[0]), float(b[1]), float(b[2]), float(b[3]), img_w, img_h
                    )
                except Exception:
                    pass

        # 2) xmin/ymin/xmax/ymax separados
        if all(k in attrs for k in ["xmin", "ymin", "xmax", "ymax"]):
            try:
                return self._to_pixel_bbox(
                    float(attrs["xmin"]), float(attrs["ymin"]),
                    float(attrs["xmax"]), float(attrs["ymax"]), img_w, img_h
                )
            except Exception:
                pass

        # 3) Quanser wrapper: x, y = centro en píxeles → construir caja estimada
        if "x" in attrs and "y" in attrs:
            try:
                cx = float(attrs["x"])
                cy = float(attrs["y"])
                name = attrs.get("name", "")

                # Seleccionar tamaño según tipo de objeto
                half_w, half_h = self._BOX_DEFAULT
                for key, (bw, bh) in self._BOX_SIZES.items():
                    if key in name.lower():
                        half_w, half_h = bw // 2, bh // 2
                        break

                return self._to_pixel_bbox(
                    cx - half_w, cy - half_h,
                    cx + half_w, cy + half_h,
                    img_w, img_h
                )
            except Exception:
                pass

        return None

    # ────────────────────────────────────────────────────────────────
    def flag_publisher(self):
        self.publish_motion_flag(self.flag_value)

    # ────────────────────────────────────────────────────────────────
    def on_timer(self):
        self._frame_count += 1

        # ── Leer cámara ─────────────────────────────────────────────
        try:
            self.QCarImg.read()
        except Exception as e:
            self.get_logger().error(f"[on_timer] ERROR en QCarImg.read(): {e}")
            return

        rgb   = self.QCarImg.rgb
        depth = self.QCarImg.depth

        if rgb is None or rgb.size == 0:
            self.get_logger().warn(f"[on_timer] frame #{self._frame_count}: rgb está VACÍO")
            return
        if depth is None or depth.size == 0:
            self.get_logger().warn(f"[on_timer] frame #{self._frame_count}: depth está VACÍO")
            return

        # ── Publicar rgb y depth ─────────────────────────────────────
        try:
            msg_rgb   = self.bridge.cv2_to_imgmsg(rgb, "bgr8")
            depth_f32 = depth.astype(np.float32)
            msg_depth = self.bridge.cv2_to_imgmsg(depth_f32, "32FC1")
            self.publish_rgb.publish(msg_rgb)
            self.publish_depth.publish(msg_depth)
        except Exception as e:
            self.get_logger().error(f"[on_timer] ERROR publicando rgb/depth: {e}")

        # ── Log periódico de estado (cada 5 s) ───────────────────────
        now = time.time()
        if now - self._last_log_t >= 5.0:
            self._last_log_t = now
            self.get_logger().info(
                f"[STATUS] frames={self._frame_count}  "
                f"detections_run={self._detect_count}  "
                f"sign_detected={self.sign_detected}  "
                f"flag_value={self.flag_value}  "
                f"speed_limit={self.speed_limit}  "
                f"rgb_shape={rgb.shape}  "
                f"depth_shape={depth.shape}  "
                f"depth_dtype={depth.dtype}"
            )

        # ── Lógica de detección ──────────────────────────────────────
        current_time = now - self.t0

        if not self.sign_detected:
            sign_delay, sign_detected = self.yolo_detect()
            delay = sign_delay if sign_detected else 0.0

            if delay > 0.0 and not self.sign_detected:
                self.sign_detected = True
                self.disable_until = delay
                self.flag_value    = False
                self.get_logger().warn(
                    f"[on_timer] Señal detectada! Deteniendo por {delay:.1f}s"
                )
            else:
                self.flag_value = True

        else:  # sign_detected == True
            if current_time >= self.disable_until:
                if current_time >= self.detection_cooldown:
                    self.sign_detected = False
                    self.get_logger().info("[on_timer] Cooldown completado, sign_detected=False")
                self.flag_value = True

    # ────────────────────────────────────────────────────────────────
    def yolo_detect(self):
        self._detect_count += 1
        detected = False
        delay    = 0.0

        # ── Pre-proceso ──────────────────────────────────────────────
        try:
            rgbProcessed = self.myYolo.pre_process(self.QCarImg.rgb)
        except Exception as e:
            self.get_logger().error(f"[yolo_detect] ERROR en pre_process: {e}")
            return 0.0, False

        # ── Inferencia ───────────────────────────────────────────────
        try:
            _ = self.myYolo.predict(
                inputImg   = rgbProcessed,
                #classes   = [2, 9, 11, 33],
                confidence = 0.3,
                half       = True,
                verbose    = False,
            )
        except Exception as e:
            self.get_logger().error(f"[yolo_detect] ERROR en predict: {e}")
            return 0.0, False

        # ── Post-proceso ─────────────────────────────────────────────
        try:
            processedResults = self.myYolo.post_processing(
                alignedDepth     = self.QCarImg.depth,
                clippingDistance = 5,
            )
        except Exception as e:
            self.get_logger().error(f"[yolo_detect] ERROR en post_processing: {e}")
            return 0.0, False

        n_results = len(processedResults) if processedResults else 0
        if self._detect_count % 30 == 1:
            self.get_logger().info(
                f"[yolo_detect] llamada #{self._detect_count}  objetos_detectados={n_results}"
            )

        annotated_img    = self.QCarImg.rgb.copy()
        h, w             = annotated_img.shape[:2]
        frame_speed_limit = 1.0   # se sobreescribe según semáforo detectado

        for obj in processedResults:
            attrs = obj.__dict__

            # Debug completo solo las primeras 3 llamadas
            if self._detect_count <= 3:
                self.get_logger().info(f"[DEBUG] objeto completo: {attrs}")

            labelName  = attrs.get("name",     "")
            labelConf  = attrs.get("conf",     0.0)
            objectDist = attrs.get("distance", -1.0)

            self.get_logger().info(
                f"[DETECTION] nombre='{labelName}'  "
                f"conf={labelConf:.3f}  "
                f"dist={objectDist:.3f}m"
            )

            # ── BBOX robusto ──────────────────────────────────────────
            bb = self._extract_bbox_xyxy(attrs, w, h)

            if bb is None:
                # Siempre logueamos keys para diagnóstico
                self.get_logger().warn(
                    f"[BBOX] '{labelName}' sin bbox. Keys: {list(attrs.keys())}"
                )
                if self.debug_keys_once:
                    self.debug_keys_once = False
                    self.get_logger().warn(f"[BBOX] attrs completo: {attrs}")
            else:
                x1, y1, x2, y2 = bb

                # Color de caja: rojo/amarillo/verde para semáforos, blanco para el resto
                light_color = attrs.get("lightColor", "").lower()
                if "red" in light_color:
                    box_color = (0, 0, 255)
                elif "yellow" in light_color:
                    box_color = (0, 255, 255)
                elif "green" in light_color:
                    box_color = (0, 255, 0)
                else:
                    box_color = (0, 255, 0)

                label_txt = f"{labelName} {labelConf:.2f} {objectDist:.2f}m"
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    annotated_img, label_txt, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2
                )

            # ── Lógica de negocio (v1 completa) ──────────────────────
            if labelName == 'car' and labelConf > 0.9 and objectDist < 0.15:
                self.get_logger().warn("[ALERT] Car found! (dist<0.45m)")

            elif labelName == "stop sign" and labelConf > 0.88 and objectDist < 0.3:
                self.get_logger().warn("[ALERT] Stop Sign Detected! Pausa 3.0s")
                delay = 3.0
                self.t0 = time.time()
                detected = True
                self.detection_cooldown = 10.0

            elif labelName == "yield sign" and labelConf > 0.9 and objectDist < 0.15:
                self.get_logger().warn("[ALERT] Yield Sign Detected! Pausa 1.5s")
                delay = 1.5
                self.t0 = time.time()
                detected = True
                self.detection_cooldown = 10.0

            # Semáforos — usan if/elif para no sobrescribirse entre sí
            if labelName == "traffic light (red )" and labelConf > 0.8:
                self.get_logger().warn("[TRAFFIC LIGHT] Red — esperando permiso para mover...")
                delay   = 0.01
                self.t0 = time.time()
                detected = True
                self.detection_cooldown = 0.01
                frame_speed_limit = 0.0

            elif labelName == "traffic light (yellow )" and labelConf > 0.8:
                self.get_logger().warn("[TRAFFIC LIGHT] Yellow — reduciendo velocidad...")
                detected          = False
                frame_speed_limit = 0.5

            elif labelName == "traffic light (green )" and labelConf > 0.8:
                self.get_logger().warn("[TRAFFIC LIGHT] Green — adelante...")
                detected          = False
                frame_speed_limit = 1.0

        # ── Publicar imagen anotada ───────────────────────────────────
        try:
            msg_det = self.bridge.cv2_to_imgmsg(annotated_img, "bgr8")
            self.publish_detections.publish(msg_det)
            if self._detect_count % 90 == 1:
                self.get_logger().info(
                    f"[yolo_detect] Imagen anotada publicada en /qcar_camera/detections "
                    f"(shape={annotated_img.shape})"
                )
        except Exception as e:
            self.get_logger().error(
                f"[yolo_detect] ERROR publicando /qcar_camera/detections: {e}"
            )

        # ── Publicar speed limit ──────────────────────────────────────
        self.speed_limit = frame_speed_limit
        speed_msg        = Float32()
        speed_msg.data   = float(self.speed_limit)
        self.publish_speed.publish(speed_msg)

        return delay, detected

    # ────────────────────────────────────────────────────────────────
    def publish_motion_flag(self, enable: bool):
        msg      = Bool()
        msg.data = enable
        self.motion_publisher.publish(msg)

    def terminate(self):
        self.get_logger().info("[terminate] Cerrando QCarImg...")
        self.QCarImg.terminate()


def main():
    rclpy.init()
    node = ObjectDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.terminate()
    rclpy.shutdown()


if __name__ == '__main__':
    main()