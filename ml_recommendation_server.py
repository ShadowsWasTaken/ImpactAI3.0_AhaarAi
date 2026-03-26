from __future__ import annotations

import io
import re
import urllib.parse
import ssl
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Bypass local SSL certificate errors
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

app = FastAPI(title="Layer3 Dual-ML Engine", version="FINAL")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Profile(BaseModel):
    age: int = Field(ge=10, le=90)
    weight: float = Field(ge=20, le=250)
    height: float = Field(ge=120, le=230)
    gender: str
    goal: str
    activity: str
    diet_preference: str

class Layer3Engine:
    def __init__(self) -> None:
        # Food ML
        self.df: Optional[pd.DataFrame] = None
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_matrix: Optional[np.ndarray] = None
        
        # Exercise ML
        self.ex_df: Optional[pd.DataFrame] = None
        self.ex_pipeline: Optional[ColumnTransformer] = None
        self.ex_feature_matrix: Optional[np.ndarray] = None
        
        self.trained: bool = False

    def _generate_food_dataset(self) -> pd.DataFrame:
        """Generates a perfectly tagged dataset of realistic meals."""
        data = []
        vegan_bases = [("Brown Rice", 150, 3, 30, 1), ("Quinoa", 120, 4, 21, 2), ("Roti", 100, 3, 20, 1), ("Oats", 130, 5, 23, 2)]
        vegan_proteins = [("Yellow Dal", 120, 9, 20, 1), ("Chole", 160, 10, 25, 3), ("Rajma", 150, 9, 22, 2), ("Tofu", 100, 10, 2, 5), ("Soya Chunks", 170, 25, 10, 1)]
        veggies = [("Spinach", 20, 2, 3, 0), ("Broccoli", 30, 2, 5, 0), ("Mixed Veg", 50, 2, 10, 0)]
        
        for b in vegan_bases:
            for p in vegan_proteins:
                for v in veggies:
                    data.append({"name": f"{b[0]} with {p[0]} and {v[0]}", "calories": b[1] + p[1] + v[1], "protein": b[2] + p[2] + v[2], "carbs": b[3] + p[3] + v[3], "fat": b[4] + p[4] + v[4], "text": f"vegan healthy {b[0]} {p[0]} {v[0]} plant based", "diet_type": "vegan"})

        veg_proteins = [("Paneer Tikka", 220, 14, 5, 15), ("Greek Yogurt", 100, 10, 4, 0), ("Palak Paneer", 240, 12, 8, 16), ("Dal Makhani", 280, 10, 35, 12)]
        for b in vegan_bases:
            for p in veg_proteins:
                for v in veggies:
                    data.append({"name": f"{b[0]} with {p[0]} and {v[0]}", "calories": b[1] + p[1] + v[1], "protein": b[2] + p[2] + v[2], "carbs": b[3] + p[3] + v[3], "fat": b[4] + p[4] + v[4], "text": f"vegetarian healthy {b[0]} {p[0]} {v[0]} dairy", "diet_type": "veg"})

        meat_proteins = [("Grilled Chicken", 165, 31, 0, 3), ("Chicken Curry", 250, 20, 10, 15), ("Fish Tikka", 200, 22, 2, 10), ("Egg Bhurji", 180, 12, 4, 12)]
        for b in vegan_bases:
            for p in meat_proteins:
                for v in veggies:
                    data.append({"name": f"{b[0]} with {p[0]} and {v[0]}", "calories": b[1] + p[1] + v[1], "protein": b[2] + p[2] + v[2], "carbs": b[3] + p[3] + v[3], "fat": b[4] + p[4] + v[4], "text": f"nonveg healthy {b[0]} {p[0]} {v[0]} meat protein", "diet_type": "nonveg"})
                    
        snacks = [("Fruit Salad & Almonds", 150, 4, 20, 8, "vegan"), ("Peanut Butter Toast", 200, 8, 20, 10, "vegan"), ("Boiled Eggs", 140, 12, 1, 10, "nonveg"), ("Whey Protein Shake", 120, 24, 3, 1, "veg"), ("Poha", 200, 4, 40, 5, "vegan"), ("Masala Oats", 150, 5, 25, 4, "vegan")]
        for s in snacks:
            data.append({"name": s[0], "calories": s[1], "protein": s[2], "carbs": s[3], "fat": s[4], "text": f"snack breakfast light {s[5]}", "diet_type": s[5]})

        return pd.DataFrame(data)

    def _generate_exercise_dataset(self) -> pd.DataFrame:
        """Secondary Dataset specifically built for the Workout ML Pipeline"""
        data = [
            # Weightloss / Cardio / Burn
            {"name": "Jump Rope", "intensity": 8, "text": "cardio full body burn weightloss high heart rate"},
            {"name": "Burpees", "intensity": 9, "text": "full body cardio burn weightloss high intensity bodyweight"},
            {"name": "Mountain Climbers", "intensity": 7, "text": "core cardio burn weightloss bodyweight"},
            {"name": "High Knees", "intensity": 8, "text": "cardio burn weightloss bodyweight legs"},
            {"name": "Kettlebell Swings", "intensity": 8, "text": "full body power burn weightloss kettlebell"},
            {"name": "Box Jumps", "intensity": 8, "text": "legs power cardio burn weightloss plyometric"},
            
            # Muscle Gain / Hypertrophy
            {"name": "Barbell Bench Press", "intensity": 9, "text": "heavy compound chest triceps hypertrophy musclegain push"},
            {"name": "Incline Dumbbell Press", "intensity": 8, "text": "chest hypertrophy musclegain upper chest push"},
            {"name": "Overhead Press", "intensity": 8, "text": "shoulders heavy compound hypertrophy musclegain push"},
            {"name": "Tricep Dips", "intensity": 7, "text": "triceps chest bodyweight hypertrophy musclegain push"},
            {"name": "Deadlift", "intensity": 10, "text": "heavy compound back hamstrings hypertrophy musclegain pull"},
            {"name": "Pullups", "intensity": 8, "text": "back lats bodyweight hypertrophy musclegain pull"},
            {"name": "Barbell Rows", "intensity": 8, "text": "back heavy compound hypertrophy musclegain pull"},
            {"name": "Lat Pulldown", "intensity": 7, "text": "back lats machine hypertrophy musclegain pull"},
            {"name": "Dumbbell Curls", "intensity": 6, "text": "isolation biceps arms hypertrophy musclegain pull"},
            {"name": "Barbell Squats", "intensity": 9, "text": "heavy compound legs quads glutes hypertrophy musclegain"},
            {"name": "Leg Press", "intensity": 8, "text": "legs quads machine hypertrophy musclegain"},
            {"name": "Romanian Deadlifts", "intensity": 8, "text": "legs hamstrings heavy hypertrophy musclegain"},
            {"name": "Bulgarian Split Squats", "intensity": 8, "text": "legs glutes unilateral hypertrophy musclegain"},
            
            # Maintenance / Core / Balanced
            {"name": "Pushups", "intensity": 6, "text": "chest bodyweight maintenance balanced push"},
            {"name": "Bodyweight Squats", "intensity": 5, "text": "legs bodyweight maintenance balanced"},
            {"name": "Walking Lunges", "intensity": 6, "text": "legs glutes bodyweight maintenance balanced"},
            {"name": "Plank", "intensity": 5, "text": "core abs bodyweight maintenance balanced"},
            {"name": "Russian Twists", "intensity": 6, "text": "core obliques bodyweight maintenance balanced"},
            {"name": "Bicycle Crunches", "intensity": 6, "text": "core abs bodyweight maintenance balanced"},
            {"name": "Glute Bridges", "intensity": 5, "text": "glutes core bodyweight maintenance balanced"}
        ]
        return pd.DataFrame(data)

    def fetch_and_train(self) -> Dict[str, object]:
        # --- 1. TRAIN FOOD ML PIPELINE ---
        df = self._generate_food_dataset()
        preprocess = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[("scaler", StandardScaler())]), ["calories", "protein", "carbs", "fat"]),
                ("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=1000), "text"),
            ]
        )
        X = preprocess.fit_transform(df)
        self.feature_matrix = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        self.df = df.reset_index(drop=True)
        self.pipeline = preprocess

        # --- 2. TRAIN EXERCISE ML PIPELINE ---
        ex_df = self._generate_exercise_dataset()
        ex_preprocess = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[("scaler", StandardScaler())]), ["intensity"]),
                ("txt", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=500), "text"),
            ]
        )
        ex_X = ex_preprocess.fit_transform(ex_df)
        self.ex_feature_matrix = ex_X.toarray() if hasattr(ex_X, "toarray") else np.asarray(ex_X)
        self.ex_df = ex_df
        self.ex_pipeline = ex_preprocess

        self.trained = True
        return {"status": "success", "food_items": len(df), "exercise_items": len(ex_df)}

    @staticmethod
    def _calc_targets(profile: Profile) -> Dict[str, float]:
        if profile.gender.upper() == "M":
            bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age + 5
        else:
            bmr = 10 * profile.weight + 6.25 * profile.height - 5 * profile.age - 161

        activity_mul = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725}.get(profile.activity, 1.375)
        tdee = bmr * activity_mul

        if profile.goal == "weightloss":
            calories = tdee - 500
            protein = profile.weight * 2.0 
            fat = calories * 0.25 / 9
        elif profile.goal == "musclegain":
            calories = tdee + 300
            protein = profile.weight * 2.2 
            fat = calories * 0.25 / 9
        else:
            calories = tdee
            protein = profile.weight * 1.6
            fat = calories * 0.28 / 9

        carbs = (calories - protein * 4 - fat * 9) / 4
        return {"calories": round(calories, 0), "protein": round(protein, 0), "carbs": round(max(carbs, 30), 0), "fat": round(max(fat, 20), 0)}

    def recommend(self, profile: Profile) -> Dict[str, object]:
        if not self.trained or self.df is None or self.ex_df is None:
            raise HTTPException(status_code=400, detail="Model not initialized. Call /initialize first.")

        daily_targets = self._calc_targets(profile)
        
        # ==========================================
        # PIPELINE A: FOOD RECOMMENDATION
        # ==========================================
        df = self.df.copy()
        if profile.diet_preference == 'veg':
            df = df[df['diet_type'].isin(['veg', 'vegan'])]
        elif profile.diet_preference == 'vegan':
            df = df[df['diet_type'] == 'vegan']

        slots = {"Breakfast": 0.25, "Lunch": 0.35, "Snack": 0.10, "Dinner": 0.30}
        plan = []
        used = set()

        for slot, frac in slots.items():
            slot_targets = {
                "calories": daily_targets["calories"] * frac,
                "protein": daily_targets["protein"] * frac,
                "carbs": daily_targets["carbs"] * frac,
                "fat": daily_targets["fat"] * frac,
            }

            target_df = pd.DataFrame([{"calories": slot_targets["calories"], "protein": slot_targets["protein"], "carbs": slot_targets["carbs"], "fat": slot_targets["fat"], "text": "healthy"}])
            target_dense = self.pipeline.transform(target_df) # type: ignore
            target_dense = target_dense.toarray() if hasattr(target_dense, "toarray") else np.asarray(target_dense)

            sims = cosine_similarity(target_dense, self.feature_matrix[df.index])[0]
            df["sim"] = sims
            
            ranked = df.sort_values("sim", ascending=False)
            for _, row in ranked.iterrows():
                if row["name"] not in used:
                    base_cal = float(row["calories"])
                    raw_multiplier = slot_targets["calories"] / base_cal if base_cal > 0 else 1.0
                    clean_multiplier = round(raw_multiplier * 2) / 2
                    if clean_multiplier < 0.5: clean_multiplier = 0.5
                    if clean_multiplier > 2.5: continue 

                    used.add(row["name"])
                    clean_name = str(row["name"]).title()
                    yt_recipe = f"https://www.youtube.com/results?search_query={urllib.parse.quote(clean_name + ' healthy recipe')}"

                    plan.append({
                        "slot": slot,
                        "name": clean_name,
                        "serving_multiplier": clean_multiplier,
                        "adjusted_calories": int(base_cal * clean_multiplier),
                        "adjusted_protein": int(float(row["protein"]) * clean_multiplier),
                        "adjusted_carbs": int(float(row["carbs"]) * clean_multiplier),
                        "adjusted_fat": int(float(row["fat"]) * clean_multiplier),
                        "recipe_link": yt_recipe
                    })
                    break

        # ==========================================
        # PIPELINE B: EXERCISE RECOMMENDATION
        # ==========================================
        if profile.goal == "weightloss":
            ex_target_text = "cardio burn full body weightloss high intensity core plyometric"
            ex_target_int = 8
        elif profile.goal == "musclegain":
            ex_target_text = "heavy compound hypertrophy musclegain barbell isolation push pull"
            ex_target_int = 9
        else:
            ex_target_text = "maintenance balanced bodyweight core light"
            ex_target_int = 5

        ex_target_df = pd.DataFrame([{"intensity": ex_target_int, "text": ex_target_text}])
        ex_target_dense = self.ex_pipeline.transform(ex_target_df) # type: ignore
        ex_target_dense = ex_target_dense.toarray() if hasattr(ex_target_dense, "toarray") else np.asarray(ex_target_dense)

        ex_sims = cosine_similarity(ex_target_dense, self.ex_feature_matrix)[0]
        ex_df_copy = self.ex_df.copy()
        ex_df_copy["sim"] = ex_sims

        # Extract the top 9 most mathematically relevant exercises for this specific user
        top_exercises = ex_df_copy.sort_values("sim", ascending=False).head(9)["name"].tolist()

        def yt(query: str) -> str:
            return f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"

        exercise_routine = [
            {
                "day": "Day 1: Primary Target", 
                "focus": f"ML Match: {profile.goal.title()} Focus", 
                "routine": f"{top_exercises[0]} (4x10), {top_exercises[1]} (3x12), {top_exercises[2]} (3x15)",
                "yt_link": yt(f"{top_exercises[0]} proper form tutorial")
            },
            {
                "day": "Day 2: Synergistic Target", 
                "focus": f"ML Match: {profile.goal.title()} Focus", 
                "routine": f"{top_exercises[3]} (4x10), {top_exercises[4]} (3x12), {top_exercises[5]} (3x15)",
                "yt_link": yt(f"{top_exercises[3]} proper form tutorial")
            },
            {
                "day": "Day 3: Conditioning Target", 
                "focus": "ML Match: Accessory Core", 
                "routine": f"{top_exercises[6]} (3x15), {top_exercises[7]} (3x15), {top_exercises[8]} (3x60s)",
                "yt_link": yt(f"{top_exercises[6]} proper form tutorial")
            }
        ]

        return {
            "tdee_targets": daily_targets, 
            "meal_plan": plan, 
            "exercise_routine": exercise_routine
        }

engine = Layer3Engine()

@app.get("/initialize")
def initialize_system() -> Dict[str, object]:
    return engine.fetch_and_train()

@app.post("/recommend")
def recommend(profile: Profile) -> Dict[str, object]:
    return engine.recommend(profile)