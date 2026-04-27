from collections import Counter, deque


class ValueStabilizer:
    def __init__(self, buffer_size=5):
        self.buffer = deque(maxlen=buffer_size)
        self.stable_value = ""

    def update(self, new_value):
        if new_value:
            self.buffer.append(new_value)
        if not self.buffer:
            return ""
        counts = Counter(self.buffer)
        self.stable_value = counts.most_common(1)[0][0]
        return self.stable_value
