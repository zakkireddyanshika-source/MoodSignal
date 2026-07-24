import joblib
from importlib.machinery import SourceFileLoader

episode_classifier = SourceFileLoader("episode_classifier", "notebooks/11_episode_classifier.py").load_module()
trajectory_detector = SourceFileLoader("episode_classifier", "notebooks/12_trajectory_detector.py").load_module()

model = joblib.load("models/baseline_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

label_names = {0: "bipolar", 1: "depression", 2: "anxiety"}
disclaimer = "\nThis is NOT a diagnosis. These are language patterns to be discussed with a healthcare professional"

def analyze_single_entry(text):
    numbers = vectorizer.transform([text])
    probabilities = model.predict_proba(numbers)[0]
    prediction = probabilities.argmax()

    print("\nMoodSignal Analysis")
    print(f"Closest Match: {label_names[prediction]} ({probabilities[prediction]:.0%} confidence)")
    print("\nAll Scores:")
    for i, name in label_names.items():
        bar = "|" * int(probabilities[i]*20)
        print(f"  {name:<12} {probabilities[i]:.0%}  {bar}")
        print(disclaimer)

def run_multiday_tracker():
    eps = []
    day = 1
    print("\nMulti-Day Mood Tracker")
    print("Type 'Done' when finished entering days\n")
    while True:
        text = input(f"Day {day} - How are you feeling? ")
        if text.lower() == 'done':
            break

        sleep = float(input("  Hours of sleep last night: "))
        energy = int(input("  Energy level (1-10): "))
        mood = int(input("  Mood rating (1-5): "))
        racing = int(input("  Racing Thoughts (0=No, 1=Yes): "))
        impulsive = int(input("  Impulsive Thoughts (0=No, 1=Yes): "))
        irritable = int(input("  Irritable (0=No, 1=Yes): "))
        htgu = int(input("  Hard to get up (1-10): "))

        ep = episode_classifier.classify_episode(
            text, sleep, energy, mood, racing, impulsive, irritable, htgu
        )
        eps.append(ep)
        print(f"  day {day} classified as: {ep}\n")
        day += 1

    if eps:
        print("\nYour Mood Timeline")
        trajectory_detector.build_mood_timeline(eps)
        patterns = trajectory_detector.detect_trajectory(eps)
        print(f"\nPatterns Detected: {patterns}")
        for msg in trajectory_detector.get_alert_message(patterns):
            print(f"Alert: {msg}")

    else:
        print("No Days Entered")

    print(disclaimer)

while True:
    print("\n--- MoodSignal ---")
    print("1. Analyze a single entry")
    print("2. Multi-Day Mood Tracker")
    print("3. Exit")

    choice = input("\nPick An Option (1/2/3): ")
    if choice == "1":
        text = input("\nType how you are feeling: ")
        analyze_single_entry(text)
    elif choice == "2":
        run_multiday_tracker()
    elif choice == "3":
        print("Bye Bye")
        break
    else:
        print("Please type 1, 2, or 3")
        