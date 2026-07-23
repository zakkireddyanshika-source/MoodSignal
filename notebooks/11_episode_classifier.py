manic_words = ["cant stop", "no sleep", "dont need sleep", "racing thoughts", "unstoppable", "amazing", "genius", "special", "extraordinary",]
depressive_words = ["hopeless", "worthless", "tired", "exhausted", "nothing", "pointless", "cant get up", "empty"]

def count_words(text, word_list):
    text = text.lower()
    return sum(1 for word in word_list if word in text)
def classify_episode(text, sleep_hours, energy_level, mood_rating, racing_thoughts, impulsive, irritable, hard_to_get_up):
    if sleep_hours < 5 and energy_level > 7 and mood_rating == 5:
        return "manic"
    elif sleep_hours < 6 and energy_level > 6 and mood_rating >= 4:
        return "hypomanic"
    elif energy_level < 4 and mood_rating <= 2 and hard_to_get_up > 6:
        return "depressive"
    else:
        return "stable"

manic_test = classify_episode("I feel incredible, havent slept in 2 days, so many ideas",
                               sleep_hours=2, energy_level=10, mood_rating=5,
                               racing_thoughts=1, impulsive=1, irritable=0, hard_to_get_up=1)
print(f"manic test: {manic_test}")

hypomanic_test = classify_episode("feeling really good, unusually productive",
                                   sleep_hours=5, energy_level=8, mood_rating=4,
                                   racing_thoughts=1, impulsive=0, irritable=0, hard_to_get_up=2)
print(f"hypomanic test: {hypomanic_test}")

depressive_test = classify_episode("cant get out of bed, everything feels pointless",
                                    sleep_hours=12, energy_level=2, mood_rating=1,
                                    racing_thoughts=0, impulsive=0, irritable=1, hard_to_get_up=9)
print(f"depressive test: {depressive_test}")

stable_test = classify_episode("had an okay day, went to work, came home",
                                sleep_hours=8, energy_level=6, mood_rating=3,
                                racing_thoughts=0, impulsive=0, irritable=0, hard_to_get_up=4)
print(f"stable test: {stable_test}")