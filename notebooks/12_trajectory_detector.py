def count_consecutive(labels, target):
    max_run = 0
    current = 0 
    for label in labels:
        if label == target:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run

def detect_trajectory(episode_list):
    patterns = []

    if count_consecutive(episode_list, "manic") >= 3:
        patterns.append("escalating_mania")
    if count_consecutive(episode_list, "depressive") >= 5:
        patterns.append("sustained_depression")

    if count_consecutive(episode_list, "stable") >= 7:
        patterns.append("stable_streak")

    changes = 0 
    for i in range(1, min(7, len(episode_list))):
        if episode_list[i] != episode_list[i-1]:
            changes += 1

    if changes >= 3:
        patterns.append("rapid_cycling")

    for i, label in enumerate(episode_list):
        if label == "depressive":
            next_few = episode_list[i + 1:i + 4]
            if "manic" in next_few or "hypomanic" in next_few:
                patterns.append("episode_switch")
                break

    if not patterns: 
        patterns.append("no_pattern")
    return patterns

def build_mood_timeline(episode_list):
    letters = {"manic": "M", "hypomanic": "H", "depressive": "D", "stable": "S"}
    for day, label in enumerate(episode_list, start=1):
        print(f"day{day}: {letters[label]} {label}")

def get_alert_message(patterns):
    messages = {"escalating_mania": "3 or more manic days in a row. worth discusing with your provider.",  "sustained_depression": "extended low mood pattern. consider reaching out to your provider.", "stable_streak": "stable mood pattern, positive sign. keep tracking.", "rapid_cycling": "frequent mood changes detected. worth mentioning to your provider.",  "episode_switch": "mood shift from low to elevated. worth tracking closely.", "no_pattern": "no significant pattern detected yet. keep tracking.",}
    return [messages[p] for p in patterns]

sequence_1 = ["stable", "stable", "hypomanic", "manic", "manic", "manic", "depressive"]
sequence_2 = ["depressive", "depressive", "depressive", "depressive", "depressive", "depressive"]
sequence_3 = ["stable", "stable", "stable", "stable", "stable", "stable", "stable", "stable"]

for sequence in [sequence_1, sequence_2, sequence_3]:
    print("\ntimeline")
    build_mood_timeline(sequence)
    patterns = detect_trajectory(sequence)
    print(f"patterns detected: {patterns}")
    for message in get_alert_message(patterns):
        print(f"alert: {message}")
