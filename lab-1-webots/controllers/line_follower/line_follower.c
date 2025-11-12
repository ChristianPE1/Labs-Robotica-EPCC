#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/distance_sensor.h>
#include <stdio.h>

#define TIME_STEP 32

// PID base (ajustables)
#define KP 2.0
#define KI 0.01
#define KD 0.35

// Anti-windup
#define I_MAX 100.0
#define I_MIN -100.0

// Velocidad base y limites
#define BASE_SPEED 4.0
#define MAX_SPEED 6.28

// Sensores de suelo (usamos 3)
#define NB_GS 3
WbDeviceTag gs[NB_GS];
double gs_value[NB_GS];

// Motores
WbDeviceTag left_motor, right_motor;

// Variables PID y filtrado
double last_error = 0.0;
double integral = 0.0;
double prev_filtered[NB_GS] = {0.0, 0.0, 0.0};

// Normalizacion: crudo -> [0,1], 1 = linea (negra), 0 = fondo (blanco)
double normalize_raw(double value) {
  const double MIN_RAW = 300.0; // ajusta segun resultados - color negro
  const double MAX_RAW = 900.0; // ajusta segun lecturas: - color blanco
  double norm = 1.0 - ((value - MIN_RAW) / (MAX_RAW - MIN_RAW)); // invertido - blanco a negro
  if (norm < 0.0) norm = 0.0;
  if (norm > 1.0) norm = 1.0;
  return norm;
}

// filtro exponencial simple para suavizar lecturas
double lowpass(double prev, double current, double alpha) {
  return prev * (1.0 - alpha) + current * alpha;
}

int main() {
  wb_robot_init();

  // inicializar sensores
  char name[10];
  for (int i = 0; i < NB_GS; i++) {
    sprintf(name, "gs%d", i);
    gs[i] = wb_robot_get_device(name);
    wb_distance_sensor_enable(gs[i], TIME_STEP);
    prev_filtered[i] = 0.0;
  }

  // inicializar motores
  left_motor = wb_robot_get_device("left wheel motor");
  right_motor = wb_robot_get_device("right wheel motor");
  wb_motor_set_position(left_motor, INFINITY);
  wb_motor_set_position(right_motor, INFINITY);
  wb_motor_set_velocity(left_motor, 0.0);
  wb_motor_set_velocity(right_motor, 0.0);

  while (wb_robot_step(TIME_STEP) != -1) {
    // leer y normalizar sensores con filtro
    for (int i = 0; i < NB_GS; i++) {
      double raw = wb_distance_sensor_get_value(gs[i]);
      double norm = normalize_raw(raw);
      // alpha pequeño -> mucha suavidad; ajustar entre 0.2 y 0.5
      prev_filtered[i] = lowpass(prev_filtered[i], norm, 0.35);
      gs_value[i] = prev_filtered[i];
    }

    double left = gs_value[0];
    double center = gs_value[1];
    double right = gs_value[2];

    // error ponderado: toma en cuenta center con peso menor para mayor estabilidad
    double error = (1.0 * left + 0.3 * center) - (1.0 * right + 0.3 * center);
    // equivale aprox left - right, pero con efecto del center

    // PID
    integral += error;
    // anti-windup
    if (integral > I_MAX) integral = I_MAX;
    if (integral < I_MIN) integral = I_MIN;
    double derivative = (error - last_error);
    double correction = KP * error + KI * integral + KD * derivative;
    last_error = error;

    // si center es pequeno (no centrado), aumentar giro (cuando esta en el borde)
    // factor en [1.0, 1.8] cuando center->0
    double adapt_gain = 1.0 + (1.0 - center) * 0.8;
    correction *= adapt_gain;

    // limitar magnitud de correction para evitar giros excesivos
    const double MAX_CORR = 6.0;
    if (correction > MAX_CORR) correction = MAX_CORR;
    if (correction < -MAX_CORR) correction = -MAX_CORR;

    // reducir velocidad base en curvas fuertes (mejor estabilidad)
    double curve_severity = fabs(correction) / MAX_CORR; // [0,1]
    double speed_scale = 1.0 - 0.5 * curve_severity; // hasta 50% reduction
    double base_speed = BASE_SPEED * speed_scale;
    if (base_speed < 1.5) base_speed = 1.5; // minimo de avance

    double left_speed = base_speed - correction;
    double right_speed = base_speed + correction;

    // cuando center detecta linea se prioriza avanzar recto
    if (center > 0.6) {
      left_speed = base_speed;
      right_speed = base_speed;
    }

    // si pierde la linea busca girando hacia ultimo error
    if (left < 0.12 && center < 0.12 && right < 0.12) {
      if (last_error > 0) {
        left_speed = -MAX_SPEED * 0.6;
        right_speed = MAX_SPEED * 0.6;
      } else {
        left_speed = MAX_SPEED * 0.6;
        right_speed = -MAX_SPEED * 0.6;
      }
    }

    // saturacion final
    if (left_speed > MAX_SPEED) left_speed = MAX_SPEED;
    if (right_speed > MAX_SPEED) right_speed = MAX_SPEED;
    if (left_speed < -MAX_SPEED) left_speed = -MAX_SPEED;
    if (right_speed < -MAX_SPEED) right_speed = -MAX_SPEED;

    // aplicar motores
    wb_motor_set_velocity(left_motor, left_speed);
    wb_motor_set_velocity(right_motor, right_speed);

    // imprime cada X iteraciones
    printf("GS:[%.2f,%.2f,%.2f] Err:%.3f Corr:%.3f adapt:%.2f base:%.2f L:%.2f R:%.2f\n",
           left, center, right, error, correction, adapt_gain, base_speed, left_speed, right_speed);
  }

  wb_robot_cleanup();
  return 0;
}
