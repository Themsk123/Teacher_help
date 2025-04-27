# main.py

from utils.feedback_handler import give_feedback
from utils.qa_generator import generate_initial_paper
from utils.note_maker import run_note_maker

# Sample config
config = {
    "exam_level": "College Level",
    "total_marks": 100,
    "num_questions": 10,
    "objective_count": 4,
    "subjective_count": 6
}

pdf_path = "attention.pdf"  # Your PDF file here
history = []  # Store feedback + generated outputs

# Step 1: Generate initial question paper
#generate_initial_paper(config, pdf_path, history)

# Step 2: Give feedback and regenerate question paper
#give_feedback("I want all objective questions first, followed by all subjective questions.", config, pdf_path, history)

video_url = "https://www.youtube.com/watch?v=_ZvnD73m40o"
num_pages = 3
session_id = "firstchat"
feedbacks = [
    "Add more real-world examples and clarify the conclusion",
    "Summarize it in simpler language for high school students"
]

results = run_note_maker(video_url, num_pages, session_id, feedbacks)

for title, content in results:
    print(title, content)