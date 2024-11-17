from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


# Define the Researcher Agent
researcher = Agent(
    role='Research Analyst',
    goal=(
        'Conduct in-depth market research on {job_title} roles at {company}. '
        'Provide insights on required skills, industry trends, and typical qualifications. '
        'Include salary ranges, geographic demand distribution, and future growth predictions for the role.'
    ),
    backstory='A highly skilled market researcher with expertise in job market analysis.',
    tools=[SerperDevTool(),ScrapeWebsiteTool()],
    verbose=True
)

# Define the Resume Evaluator Agent
resume_evaluator = Agent(
    role='Resume Evaluator',
    goal=(
        'Provide a comprehensive evaluation of the candidate\'s resume for the {job_title} role at {company}. '
        'Assess alignment with the job description, company culture, and required skills.'
    ),
    backstory='An experienced HR professional specializing in personalized resume reviews.',
    verbose=True,
    strategy=(
        '''
        Framework:
        1. **Relevance to Job Title**: Alignment with {job_title} requirements.
        2. **Company Fit**: Cultural and value alignment with {company}.
        3. **Skills Match**: Comparison of critical skills.
        4. **Experience**: Analysis of past roles and achievements.
        5. **Presentation**: Formatting and readability.
        '''
    )
)

# **New Agent 1**: Interview Coach Agent
interview_coach = Agent(
    role='Interview Coach',
    goal=(
        'Prepare the candidate for interviews at {company} for the {job_title} role. '
        'Provide insights on potential questions, company culture, and best practices.'
    ),
    backstory='A seasoned HR specialist focusing on interview coaching and mock sessions.',
    verbose=True,
    strategy=(
        '''
        1. **Common Questions**: Based on {job_title} and {company}.
        2. **Behavioral Questions**: STAR framework preparation.
        3. **Cultural Fit**: Understanding {company} values.
        4. **Negotiation Tips**: Salary expectations and benefits.
        '''
    )
)

# **New Agent 2**: Job Recommender Agent
job_recommender = Agent(
    role='Job Recommender',
    goal=(
        'Identify alternative job opportunities for the candidate based on their skills and experience. '
        'Recommend roles with high market demand and growth potential.'
    ),
    backstory='A career consultant skilled in finding the best-fit opportunities for job seekers.',
    tools=[SerperDevTool()    ],
    verbose=True,
    strategy=(
        '''
        1. **Profile Analysis**: Match the candidate’s skills and experience.
        2. **Market Trends**: Focus on roles with high demand.
        3. **Role Diversification**: Suggest related job titles.
        '''
    )
)

# Define Tasks
research_task = Task(
    description='Conduct detailed market research for the {job_title} role at {company}.',
    expected_output='Detailed market trends, salary ranges, and competitor analysis for {job_title} at {company}.',
    agent=researcher
)

evaluation_task = Task(
    description='Evaluate the resume against the {job_title} role at {company}.',
    expected_output='Category-wise scores, strengths, weaknesses, and an improvement plan for the resume.',
    agent=resume_evaluator
)

# **New Task 1**: Interview Preparation Task
interview_task = Task(
    description='Prepare the candidate for interviews at {company} for the {job_title} role.',
    expected_output=(
        'A guide with:\n'
        '- Common interview questions\n'
        '- Behavioral question strategies (STAR framework)\n'
        '- Insights on company culture and values\n'
        '- Salary negotiation tips.'
    ),
    agent=interview_coach
)

# **New Task 2**: Job Recommendation Task
recommendation_task = Task(
    description=(
        'Suggest alternative job opportunities that align with the candidate’s profile and are relevant to their location ({location}). '
        'Focus on roles with high growth potential in the specified region.'
    ),
    expected_output=(
        'A list of recommended job titles with:\n'
        '- Brief descriptions\n'
        '- Growth potential\n'
        '- Location-specific demand analysis\n'
        '- Company suggestions that are actively hiring in {location}'
    ),
    agent=job_recommender
)

# Create the Crew
crew = Crew(
    agents=[researcher, resume_evaluator, interview_coach, job_recommender],
    tasks=[research_task, evaluation_task, interview_task, recommendation_task],
    process=Process.sequential,
    verbose=True
)
