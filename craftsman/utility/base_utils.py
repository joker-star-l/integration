
def merge_intervals(intervals: list[tuple], distributions: list[int]=None):
    # Convert an interval list to a set to de-duplicate it, then convert it back to a list
    unique_intervals = list(set(intervals)) 
    # Sorting intervals by their starting point
    unique_intervals.sort(key=lambda x: x[0])
    unique_intervals = [list(interval) for interval in unique_intervals]
    # Merge intervals with [left, right) interval
    if not unique_intervals:
        return []
    merged_intervals = [unique_intervals[0]]
    if distributions is not None:
        merged_distributions = [distributions[0]]
    for idx, current in enumerate(unique_intervals[1:]):
        prev = merged_intervals[-1]
        # If the start of the current interval is less than or equal to the end of the previous interval
        if current[0] <= prev[1]:  
            prev[1] = max(prev[1], current[1])  # merged interval
            if distributions is not None:
                merged_distributions[-1] += distributions[idx+1]
        else:
            merged_intervals.append(current)  # Otherwise, add a new interval
            if distributions is not None:
                merged_distributions.append(distributions[idx+1])
    if distributions is not None:
        return merged_intervals, merged_distributions        
    return merged_intervals