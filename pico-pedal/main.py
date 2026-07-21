import rp2
import machine
import bluetooth
import time
import struct

# --- BLE MIDI Configuration ---
MIDI_SERVICE_UUID = bluetooth.UUID('03B80E5A-EDE8-4B33-A751-6CE34EC4C700')
MIDI_CHAR_UUID = bluetooth.UUID('7772E5DB-3868-4112-A1A9-F2669D106BF3')

class BLEMIDI:
    def __init__(self, ble, name="BHU1's guitar Pedal"):
        self.ble = ble
        self.ble.active(True)
        self.ble.irq(self.ble_irq)
        
        self.register()
        self.advertiser(name)
        self.connected = False
        self.conn_handle = None

    def ble_irq(self, event, data):
        if event == 1: 
            self.conn_handle = data[0] 
            print(f"Connected to Main Amp (Handle: {self.conn_handle})")
            self.connected = True
        elif event == 2: 
            print("Disconnected.")
            self.connected = False
            self.conn_handle = None
            self.advertiser("BHU1's guitar Pedal")

    def register(self):
        char_flags = bluetooth.FLAG_READ | bluetooth.FLAG_WRITE | bluetooth.FLAG_NOTIFY | bluetooth.FLAG_WRITE_NO_RESPONSE
        midi_char = (MIDI_CHAR_UUID, char_flags)
        midi_service = (MIDI_SERVICE_UUID, (midi_char,),)
        
        services = (midi_service,)
        ((self.char_handle,),) = self.ble.gatts_register_services(services)

    def advertiser(self, name):
        payload = bytearray([0x02, 0x01, 0x06])
        uuid_bytes = bytes([
            0x00, 0xC7, 0xC4, 0x4E, 0xE3, 0x6C, 0x51, 0xA7, 
            0x33, 0x4B, 0xE8, 0xED, 0x5A, 0x0E, 0xB8, 0x03
        ])
        payload += bytearray([17, 0x07]) + uuid_bytes
        
        name_bytes = name.encode('utf-8')
        scan_resp = bytearray([len(name_bytes) + 1, 0x09]) + name_bytes
        
        self.ble.gap_advertise(100000, adv_data=payload, resp_data=scan_resp)
        print(f"Advertising as BLE MIDI (Name: {name})... Waiting for connection.")

    def send_midi(self, status, data1, data2):
        if not self.connected or self.conn_handle is None:
            return
            
        header = 0x80
        timestamp = 0x80
        
        packet = struct.pack("BBBBB", header, timestamp, status, data1, data2)
        self.ble.gatts_notify(self.conn_handle, self.char_handle, packet)

# --- Setup Buzzer ---
buzzer = machine.PWM(machine.Pin(0))
buzzer.duty_u16(0) 

def beep(pitch):
    buzzer.freq(pitch) 
    buzzer.duty_u16(32768) 
    time.sleep_ms(50) 
    buzzer.duty_u16(0) 

# --- Main Loop ---
ble = bluetooth.BLE()
midi = BLEMIDI(ble)

last_state = rp2.bootsel_button()

while True:
    current_state = rp2.bootsel_button()
    
    if current_state != last_state:
        if current_state == 1: 
            # Send 127 EVERY time you press it down
            print("BOOTSEL Pressed - Triggering Toggle (127)")
            midi.send_midi(0xB0, 80, 127) 
            beep(1000) 
        else:
            # Send 0 EVERY time you let go
            print("BOOTSEL Released - Resetting (0)")
            midi.send_midi(0xB0, 80, 0)
            
        last_state = current_state
        time.sleep_ms(50) # Debounce delay
        
    time.sleep_ms(10)
