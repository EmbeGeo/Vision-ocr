import json
import logging
import threading
import time
import urllib.request
from datetime import datetime


class DataSender:
    """OCR 판독 결과를 주기적으로 외부 서버로 전송하는 클래스 (데이터 검증 포함)"""
    def __init__(self, target_ip, target_port=8000, target_path="/ocr", results_dict=None, interval=10):
        self.url = f"http://{target_ip}:{target_port}{target_path}"
        self.results_dict = results_dict
        self.interval = interval
        self.stopped = False

        self.valid_history = {}
        self.MAX_VAL = 500.0
        self.MAX_DELTA = 10.0

    def start(self):
        self.thread = threading.Thread(target=self._run_sender, daemon=True)
        self.thread.start()
        logging.info(f"[Sender] Data export started (Target: {self.url}, Interval: {self.interval}s)")

    def validate_and_filter(self):
        validated_results = {}
        for var_name, raw_val in self.results_dict.items():
            try:
                current_val = float(raw_val.replace(',', ''))

                if current_val > self.MAX_VAL:
                    logging.warning(f"[Validation] {var_name}: Value {current_val} too high (> {self.MAX_VAL}). Ignored.")
                    if var_name in self.valid_history:
                        validated_results[var_name] = str(self.valid_history[var_name])
                    continue

                if var_name in self.valid_history:
                    delta = abs(current_val - self.valid_history[var_name])
                    if delta > self.MAX_DELTA:
                        logging.warning(f"[Validation] {var_name}: Sudden jump ({self.valid_history[var_name]} -> {current_val}). Using previous value.")
                        validated_results[var_name] = str(self.valid_history[var_name])
                        continue

                self.valid_history[var_name] = current_val
                validated_results[var_name] = str(current_val)

            except (ValueError, TypeError):
                if var_name in self.valid_history:
                    validated_results[var_name] = str(self.valid_history[var_name])
                continue

        return validated_results

    def _run_sender(self):
        while not self.stopped:
            time.sleep(self.interval)
            if not self.results_dict:
                continue
            try:
                processed_results = self.validate_and_filter()
                if not processed_results:
                    continue

                payload = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "active",
                    "results": processed_results,
                }

                json_data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(
                    self.url, data=json_data, headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=3) as response:
                    if response.status == 200:
                        logging.debug(f"[Sender] Data sent successfully at {payload['timestamp']}")
            except Exception as e:
                logging.error(f"[Sender] Failed to send data: {e}")

    def stop(self):
        self.stopped = True
