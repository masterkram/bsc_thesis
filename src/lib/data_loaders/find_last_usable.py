def find_last_usable():
    lastPrediction = self.radar_files[(self.rad_len - 1) - (self.radar_seq_len - 1)]
    lastInput = find_matching_string(self.satellite_files, lastPrediction)
    if lastInput is None:
        lastPrediction = self.radar_files[
            (self.rad_len - 1) - (self.radar_seq_len - 1) - 1
        ]
        lastInput = find_matching_string(self.satellite_files, lastPrediction)
