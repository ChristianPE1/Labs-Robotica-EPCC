from controller import Robot
import matplotlib.pyplot as plt
import numpy as np

class LidarMapper(Robot):
    def __init__(self):
        super().__init__()
        self.timeStep = 64

        # --- Sensores ---
        self.lidar = self.getDevice('lidar')
        self.lidar.enable(self.timeStep)
        self.lidar.enablePointCloud()

        self.us0 = self.getDevice('us0')
        self.us1 = self.getDevice('us1')
        self.us0.enable(self.timeStep)
        self.us1.enable(self.timeStep)

        # --- Motores ---
        self.left_motor = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # --- Configuración movimiento ---
        self.coefficients = [[12.0, -6.0], [-10.0, 8.0]]
        self.base_speed = 6.0
        self.speed = [0, 0]
        self.us_value = [0, 0]

        # --- Configuración del mapa ---
        plt.ion()  # modo interactivo
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.scatter = self.ax.scatter([], [], s=2, c='blue')
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.set_title('Mapa LIDAR en tiempo real')
        self.robot_dot, = self.ax.plot(0, 0, 'ro', markersize=6)  # punto rojo = robot

    def update_plot(self, x_vals, y_vals):
        # Actualiza el gráfico en tiempo real.
        self.scatter.set_offsets(np.c_[x_vals, y_vals])
        plt.draw()
        plt.pause(0.001)

    def run(self):
        # Bucle principal del controlador
        while self.step(self.timeStep) != -1:
            # --- Movimiento del robot ---
            self.us_value[0] = self.us0.getValue()
            self.us_value[1] = self.us1.getValue()
            for i in range(2):
                self.speed[i] = 0
                for k in range(2):
                    self.speed[i] += self.us_value[k] * self.coefficients[i][k]

            self.left_motor.setVelocity(self.base_speed + self.speed[0])
            self.right_motor.setVelocity(self.base_speed + self.speed[1])

            # --- Lectura del LIDAR ---
            point_cloud = self.lidar.getPointCloud()
            x_vals, y_vals = [], []

            for point in point_cloud:
                # Se dibuja en 2D (vista superior)
                x_vals.append(point.x)
                y_vals.append(point.y)

            # --- Detección de obstáculos cercanos ---
            distances = np.sqrt(np.array(x_vals)**2 + np.array(y_vals)**2)
            if np.any(distances < 0.25):  # obstáculo a menos de 25 cm
                self.ax.set_title('[X] Obstáculo detectado')
            else:
                self.ax.set_title('Mapa LIDAR en tiempo real')

            # --- Actualiza el mapa ---
            self.update_plot(x_vals, y_vals)


controller = LidarMapper()
controller.run()

