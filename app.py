import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

df_course = pd.read_csv("coursera_course_dataset.csv", index_col=0).reset_index(drop=True)
df_course = df_course[['Title', 'Skills']]

vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(', '))
course_skills_matrix = vectorizer.fit_transform(df_course['Skills'])

def recommend_courses(desired_skills: str):
    user_skills_matrix = vectorizer.transform([desired_skills])
    similarities = cosine_similarity(user_skills_matrix, course_skills_matrix)
    recommendations = pd.DataFrame(similarities, columns=df_course['Title'])
    top_courses = recommendations.iloc[0].nlargest(3).index.tolist()
    return top_courses
class UserSkills(BaseModel):
    skills: str
@app.post("/recommendations/")
async def get_recommendations(user_skills: UserSkills):
    recommended_courses = recommend_courses(user_skills.skills)
    return {"recommended_courses": recommended_courses}

# uvicorn app:app --reload
'''
Sample Input :

{
  "skills": "Web Development, HTML, CSS, JavaScript"
}


'''