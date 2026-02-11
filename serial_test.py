import serial
import time
data = serial.Serial(
                    'COM3',
                    baudrate = 9600,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                    timeout=5
                    )

def Send(a):
  data.write(str.encode(a))
  print(f'sent-------->{a}')

# Send("A")