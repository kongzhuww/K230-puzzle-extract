"""
UART 通信协议（K230 → MSPM0G）
格式: $CMD,data*XX\n  其中 XX 为 $ 到 * 之间的 XOR 校验(2位hex)
"""

from machine import UART, FPIOA

UART_ID = 1
UART_BAUD = 115200
UART_TX_PIN = 38  # TODO: 确认引脚
UART_RX_PIN = 39  # TODO: 确认引脚


def _xor_checksum(data: str) -> str:
    cs = 0
    for ch in data:
        cs ^= ord(ch)
    return "%02X" % (cs & 0xFF)


def _make_frame(body: str) -> str:
    return "$%s*%s\n" % (body, _xor_checksum(body))


class Protocol:
    def __init__(self):
        fpioa = FPIOA()
        fpioa.set_function(UART_TX_PIN, FPIOA.UART1_TXD)
        fpioa.set_function(UART_RX_PIN, FPIOA.UART1_RXD)
        self.uart = UART(UART_ID, baudrate=UART_BAUD)

    def send_start(self):
        self.uart.write(_make_frame("START"))

    def send_end(self):
        self.uart.write(_make_frame("END"))

    def send_move(self, piece_id, cur_x_mm, cur_y_mm, cur_ang_01deg,
                  tgt_x_mm, tgt_y_mm, tgt_ang_01deg):
        """发送单片移动指令。坐标mm整数，角度0.1度整数(0~3599)"""
        body = "MOVE,%d,%d,%d,%d,%d,%d,%d" % (
            piece_id,
            int(cur_x_mm), int(cur_y_mm), int(cur_ang_01deg),
            int(tgt_x_mm), int(tgt_y_mm), int(tgt_ang_01deg),
        )
        self.uart.write(_make_frame(body))

    def send_all(self, moves):
        """moves: list of dict {id, cur_x, cur_y, cur_ang, tgt_x, tgt_y, tgt_ang}
        坐标单位mm，角度单位0.1度"""
        self.send_start()
        for m in moves:
            self.send_move(
                m["id"],
                m["cur_x"], m["cur_y"], m["cur_ang"],
                m["tgt_x"], m["tgt_y"], m["tgt_ang"],
            )
        self.send_end()

    def wait_ack(self, timeout_ms=2000):
        """等待 MSPM0G 回复 $ACK 或 $NAK"""
        import time
        start = time.ticks_ms()
        buf = b""
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            data = self.uart.read(64)
            if data:
                buf += data
                if b"\n" in buf:
                    line = buf.decode().strip()
                    if "ACK" in line:
                        return True
                    if "NAK" in line:
                        return False
            time.sleep_ms(10)
        return None
