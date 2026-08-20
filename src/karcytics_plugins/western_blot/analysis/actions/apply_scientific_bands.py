# karcytics_plugins/western_blot/analysis/actions/apply_scientific_bands.py

import numpy as np

from ..peak_analysis import DetectedBand


class ApplyScientificBandsAction:
    def __init__(self, state, lane_types: dict, tolerance_px: float = 12.0):
        self.state = state
        self.lane_types = lane_types
        self.tolerance_px = tolerance_px

    def execute(self):
        if not self.state.lanes or not self.state.bands:
            return

        valid_lanes = [
            lane.index
            for lane in self.state.lanes
            if self.lane_types.get(lane.index, "Sample")
            not in ["Ladder", "Exclude", "Unmapped", "None"]
        ]

        if not valid_lanes:
            return

        valid_bands = []
        preserved_bands = []
        for b in self.state.bands:
            if b.lane_index in valid_lanes:
                valid_bands.append(b)
            else:
                preserved_bands.append(b)

        if not valid_bands:
            self.state.bands = preserved_bands
            return

        valid_bands.sort(key=lambda b: b.position)

        # 1. ROBUST CLUSTERING: Nearest Valid Center Assignment
        clusters = []  # List of dicts: {lane_index: band}

        for b in valid_bands:
            best_cluster = None
            best_dist = float("inf")

            for cluster in clusters:
                # RULE 1: A cluster cannot have two bands from the same lane. Period.
                if b.lane_index in cluster:
                    continue

                # RULE 2: Calculate the center of the cluster. Manual anchors pull it tightly.
                manuals = [cb for cb in cluster.values() if getattr(cb, "snr", 0) >= 9.9]
                if manuals:
                    center = np.mean([cb.position for cb in manuals])
                else:
                    center = np.median([cb.position for cb in cluster.values()])

                dist = abs(b.position - center)

                # RULE 3: Must be within tolerance, and we pick the CLOSEST cluster
                if dist <= self.tolerance_px and dist < best_dist:
                    best_dist = dist
                    best_cluster = cluster

            if best_cluster is not None:
                best_cluster[b.lane_index] = b
            else:
                # Start a brand new cluster
                clusters.append({b.lane_index: b})

        # 2. Build consensus and backfill
        scientific_bands = []
        for match_id, cluster_dict in enumerate(clusters):
            cluster_bands = list(cluster_dict.values())

            manual_bands = [b for b in cluster_bands if getattr(b, "snr", 0) >= 9.9]
            if manual_bands:
                # Manual Anchors win
                pos = int(round(np.mean([b.position for b in manual_bands])))
                width = float(np.mean([b.width for b in manual_bands]))
            else:
                # Standard median
                pos = int(round(np.median([b.position for b in cluster_bands])))
                width = float(np.median([b.width for b in cluster_bands]))

            for lane_idx in valid_lanes:
                original = cluster_dict.get(lane_idx)
                if original:
                    original.position = pos
                    original.width = width
                    original.matched_band = match_id
                    scientific_bands.append(original)
                else:
                    # Backfill missing bands
                    new_band = DetectedBand(
                        lane_index=lane_idx,
                        band_index=0,
                        position=pos,
                        width=width,
                        peak_height=0.05,
                        raw_height=0.0,
                        integrated_intensity=0.05,
                        matched_band=match_id,
                    )
                    scientific_bands.append(new_band)

        # 3. Recombine, sort, and update indices
        final_bands = preserved_bands + scientific_bands
        final_bands.sort(key=lambda b: (b.lane_index, b.position))

        current_lane = -1
        b_idx = 0
        for b in final_bands:
            if b.lane_index != current_lane:
                current_lane = b.lane_index
                b_idx = 0
            b.band_index = b_idx
            b_idx += 1

        self.state.bands = final_bands
