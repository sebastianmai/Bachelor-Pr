import board
import adafruit_tcs34725

class RGB_TCS34725():

    def __init__(self):
        self.i2c = board.I2C()
        self.sensor = adafruit_tcs34725.TCS34725(self.i2c)

    def getData(self):
        rgb = self.sensor.color_rgb_bytes
        red = rgb[0]
        green = rgb[1]
        blue = rgb[2]

        col_temp = self.sensor.color_temperature

        intens = self.sensor.lux

        return [red, green, blue, col_temp, intens]