import csv
import math
import random

random.seed(42)

teams = [
    "Argentina", "Brazil", "France", "England", "Spain", "Germany", "Portugal", "Belgium",
    "Netherlands", "Italy", "Uruguay", "Croatia", "Mexico", "USA", "Japan", "Morocco",
    "Senegal", "South Korea", "Australia", "Canada"
]

team_stats = {}
for i, team in enumerate(teams, start=1):
    fifa_rank = i
    elo = 2100 - i * 20 + random.randint(-15, 15)
    avg_goals = round(random.uniform(1.0, 2.1), 2)
    avg_conceded = round(random.uniform(0.7, 1.6), 2)
    form = random.randint(2, 5)
    team_stats[team] = {
        "fifa_rank": fifa_rank,
        "elo": elo,
        "goals_scored_avg": avg_goals,
        "goals_conceded_avg": avg_conceded,
        "form": form,
        "average_age": round(random.uniform(25.5, 29.2), 1),
        "market_value": round(random.uniform(300, 1100), 1),
    }

rows = []
for match_id in range(1, 101):
    home, away = random.sample(teams, 2)
    home_stats = team_stats[home]
    away_stats = team_stats[away]

    rank_diff = away_stats["fifa_rank"] - home_stats["fifa_rank"]
    elo_diff = (home_stats["elo"] - away_stats["elo"]) / 100
    form_diff = home_stats["form"] - away_stats["form"]
    base = 0.25 + 0.05 * elo_diff + 0.02 * form_diff - 0.01 * rank_diff
    draw_base = 0.22 + 0.04 * math.exp(-abs(elo_diff))
    home_prob = max(0.15, min(0.75, base))
    away_prob = max(0.15, min(0.75, 1 - draw_base - home_prob))
    if home_prob + away_prob > 0.9:
        draw_base = 1 - home_prob - away_prob
    elif home_prob + away_prob < 0.6:
        away_prob = 0.6 - home_prob
    draw_prob = max(0.1, min(0.35, draw_base))

    rnd = random.random()
    if rnd < home_prob:
        result = 1
        goals_home = max(1, int(round(home_stats["goals_scored_avg"] + random.gauss(0, 0.9))))
        goals_away = max(0, int(round(away_stats["goals_conceded_avg"] + random.gauss(0, 0.8))))
    elif rnd < home_prob + draw_prob:
        result = 0
        goals_home = max(0, int(round((home_stats["goals_scored_avg"] + away_stats["goals_conceded_avg"]) / 2 + random.gauss(0, 0.6))))
        goals_away = goals_home
    else:
        result = 2
        goals_away = max(1, int(round(away_stats["goals_scored_avg"] + random.gauss(0, 0.9))))
        goals_home = max(0, int(round(home_stats["goals_conceded_avg"] + random.gauss(0, 0.8))))

    rows.append(
        {
            "team_home": home,
            "team_away": away,
            "goals_home": goals_home,
            "goals_away": goals_away,
            "fifa_rank_home": home_stats["fifa_rank"],
            "fifa_rank_away": away_stats["fifa_rank"],
            "elo_home": home_stats["elo"],
            "elo_away": away_stats["elo"],
            "form_home": home_stats["form"],
            "form_away": away_stats["form"],
            "goals_scored_avg_home": home_stats["goals_scored_avg"],
            "goals_scored_avg_away": away_stats["goals_scored_avg"],
            "goals_conceded_avg_home": home_stats["goals_conceded_avg"],
            "goals_conceded_avg_away": away_stats["goals_conceded_avg"],
            "home_average_age": home_stats["average_age"],
            "away_average_age": away_stats["average_age"],
            "home_market_value": home_stats["market_value"],
            "away_market_value": away_stats["market_value"],
            "result": result,
        }
    )

with open("data/raw/matches.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "team_home",
            "team_away",
            "goals_home",
            "goals_away",
            "fifa_rank_home",
            "fifa_rank_away",
            "elo_home",
            "elo_away",
            "form_home",
            "form_away",
            "goals_scored_avg_home",
            "goals_scored_avg_away",
            "goals_conceded_avg_home",
            "goals_conceded_avg_away",
            "home_average_age",
            "away_average_age",
            "home_market_value",
            "away_market_value",
            "result",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Generado data/raw/matches.csv con {len(rows)} partidos.")
