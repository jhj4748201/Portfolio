import serial
import time

class ArduinoController:
    def __init__(self, port, baudrate):
        '''
        setup arduino
        input is port and baudrate
        if it is not connected, return "No connection with Arduino"
        '''
        self.is_arudino_connected = False
        self.port = port
        self.baudrate = baudrate
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            self.is_arudino_connected = True
        except serial.SerialException:
            print("No connection with Arduino")
    def is_connected(self):
        '''
        check arduino is connected
        '''
        return self.is_arudino_connected
    def send_start_signal(self):
        '''
        send start signal
        '''
        self.ser.write(b"START\n")
        print("Sent START signal to Arduino")

    def send_stop_signal(self):
        '''
        send stop signal
        '''
        self.ser.write(b"STOP\n")
        print("Sent STOP signal to Arduino")

    def send_signal(self, value):
        '''
        send data to arduino
        input is string type value
        add "\n" last
        '''
        value = value+"\n"
        self.ser.write(value.encode())
        print(f"Sent {value.strip()} to Arduino")

    def read_response(self):
        '''
        read response from arduino
        if there no response, return None
        if there response data, return data in str
        '''
        if self.ser.in_waiting > 0:
            response = self.ser.readline().decode('utf-8').rstrip()
            print(f"Arduino response: {response}")
            return response
        return None

    def close(self):
        '''
        close arduino
        '''
        self.ser.close()
