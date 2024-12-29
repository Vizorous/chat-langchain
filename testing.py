from gradio_client import Client, handle_file
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import copy
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import requests
from datetime import datetime
from langchain.chains.combine_documents import create_stuff_documents_chain
import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
def pdf_to_chunks(pdf):
    #get pdf from file name
    # with open(pdf, "wb") as f: 
      pdf_reader = PdfReader(pdf)

      # extrat text from each page separately
      text = ""
      for page in pdf_reader.pages:
          text += page.extract_text()

      # Split the long text into small chunks.
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=700,
          chunk_overlap=200,
          length_function=len)

      chunks = text_splitter.split_text(text=text)
      return chunks

def download_pdf_from_url(url):
    response = requests.get(url,stream=True)
    if response.status_code == 200:
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"temp_{current_time}.pdf"
        with open(file_name, "wb") as f:
          for chunk in response.iter_content(2000):
            f.write(chunk)
        # get pdf file location

        return file_name
    else:
        raise Exception("Failed to download PDF")


def summary_prompt(query_with_chunks):

    query = f''' need to detailed summarization of below resume and finally conclude them. Make sure the output is within 20% of chat gpt 4o input context window.

                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                {query_with_chunks}
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return query


def analyze_cv(chunks, analyze,prompt):
        if not os.environ.get("OPENAI_API_KEY"):
          os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

        vectorstores = FAISS.from_texts(chunks, embedding=embeddings)

        # compares the query and chunks, enabling the selection of the top 'K' most similar chunks based on their similarity scores.
        docs = vectorstores.similarity_search(query=analyze, k=3)

        # creates an OpenAI object, using the ChatGPT 3.5 Turbo model
        llm = ChatOpenAI(model='gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))

        # question-answering (QA) pipeline, making use of the load_qa_chain function
        chain = create_stuff_documents_chain(llm, prompt)

        # response = chain.run(input_documents=docs, question=analyze)
        response=chain.invoke({"context": docs})
        # print(response)
        return response

def get_summary(chunks):

    summary_prompt_output = summary_prompt(query_with_chunks=chunks)
    prompt = ChatPromptTemplate.from_messages(
           ["need to detailed summarization of below resume and finally conclude them. Make sure the output is within 20% of chat gpt 4o input context window.\n\n{context}\n"]
        )
    summary = analyze_cv(chunks=chunks, analyze=summary_prompt_output,prompt=prompt)
    return summary
  
def strength_prompt(query_with_chunks):
    query = f'''need to detailed analysis and explain of the strength of below resume and finally conclude them. Make sure the output is within 20% of chat gpt 4o input context window.
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                {query_with_chunks}
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return query

def get_strengths_from_summary(chunks,summary):
    strength_prompt_output = strength_prompt(query_with_chunks=summary)
    prompt = ChatPromptTemplate.from_messages(
        ["need to detailed analysis and explain of the strength of below resume and finally conclude them. Make sure the output is within 20% of chat gpt 4o input context window.\n\n{context}\n"]
    )
    strengths = analyze_cv(chunks=chunks, analyze=strength_prompt_output,prompt=prompt)
    return strengths
def weakness_prompt(query_with_chunks):
    query = f'''need a detailed analyze and explain of the weakness of below resume and how to improve make a better resume. Make sure the output is within 20% of chat gpt 4o input context window.

                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                {query_with_chunks}
                """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                '''
    return query

def get_weaknesses_from_summary(chunks,summary):
    weakness_prompt_output = weakness_prompt(query_with_chunks=summary)
    prompt = ChatPromptTemplate.from_messages(
        ["need a detailed analyze and explain of the weakness of below resume and how to improve make a better resume. Make sure the output is within 20% of chat gpt 4o input context window.\n\n{context}\n"] )
    weaknesses = analyze_cv(chunks=chunks, analyze=weakness_prompt_output,prompt=prompt)
    return weaknesses

@tool
def getSummaryStrengthsWeaknessesFromCV(url:str) -> dict:
  """Extracts the summary, strengths and weaknesses from a CV and output it in a dict in a json like format {"summary":"summary here", "strengths:"strengths here","weaknesses":"weaknesses here"}.
  Args:
    url (str): The URL of the CV to analyze.
  Returns:
    dict: A dictionary containing the summary, strengths and weaknesses in a json like format.
    
  """
  pdf_path = download_pdf_from_url(url)
  chunks = pdf_to_chunks(pdf_path)
  summary = get_summary(chunks)
  strengths = get_strengths_from_summary(chunks,summary)
  weaknesses = get_weaknesses_from_summary(chunks,summary)
  
  return {"summary": summary, "strengths": strengths, "weaknesses": weaknesses}


@tool
def predictBig5PersonalityMBTIandProfessionalResponsibilityFromVideo(url:str) -> dict:
  """Predicts the values for Big 5 personality types, MBTI and Professional Responsibility scores for different professions from a video. outputs it in a dict in a json like format {"Big5":{"Openness":0.5, "Conscientiousness":0.5,"Extraversion":0.5,"Agreeableness":0.5,"Non-Neuroticism":0.5},"MBTI":{"type":"INFJ","score":41,0},"ProfessionalResponsibilitiesScores":{"Managers/executives":50,"Entrepreneurship":40,"Public sector professions":60,"Social/Non profit making professions":30, "Scientists/researchers, and engineers":70}}.
  Args:
    url (str): The URL of the video to analyze.
  Returns:
    dict: A dictionary in a json like format {"Big5":{"Openness":0.5, "Conscientiousness":0.5,"Extraversion":0.5,"Agreeableness":0.5,"Non-Neuroticism":0.5},"MBTI":{"type":"INFJ","score":41,0},"ProfessionalResponsibilitiesScores":{"Managers/executives":50,"Entrepreneurship":40,"Public sector professions":60,"Social/Non profit making professions":30, "Scientists/researchers, and engineers":70}}.
    
  """
  output = {}
  # return f"Here are the Big 5 Personality data",{"Openness": 0.4300081729888916, "Conscientiousness": 0.5132606029510498, "Extraversion": 0.4318406283855438, "Agreeableness": 0.5540031790733337, "Non-Neuroticism": 0.3989406228065491}
  client = Client("vizorous16/OCEANAI",verbose=True,)
  videoPredict= client.predict(
    
    language="English",
    type_modes="Files",
    files=[handle_file(url)],
    video={"video":handle_file(url),"subtitles":None},
    api_name="/event_handler_calculate_pt_scores_blocks"
  )
  for obj in videoPredict:
      if isinstance(obj, dict) and obj.get('headers') == ['Person ID', 'Path', 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Non-Neuroticism']:
          # Extract the 'data' value
          headers = obj.get('headers', None)
          data = obj.get('value', {}).get('data', None)
          break
  # create a dictionary with the extracted data, using the headers as keys
  passableData=copy.deepcopy(data)
  data = dict(zip(headers, data[0]))
  # remove the 'Person ID' and 'Path' keys
  data.pop('Person ID', None)
  data.pop('Path', None)
  output["Big5"] = data
  client.predict(
    language="English",
    type_modes="Files",
    practical_tasks="Ranking potential candidates by professional responsibilities",
    practical_subtasks="16 Personality Types of MBTI",
    practical_subtasks_selected={"Ranking potential candidates by professional responsibilities":"Professional groups","Forming effective work teams":"Finding a suitable junior colleague","Predicting consumer preferences for industrial goods":"Car characteristics","Ранжирование потенциальных кандидатов по профессиональным обязанностям":"16 персональных типов личности MBTI","Формирование эффективных рабочих коллективов":"Поиск подходящего младшего коллеги","Прогнозирование потребительских предпочтений в отношении промышленных товаров":"Характеристики автомобиля"},
    api_name="/event_handler_practical_subtasks"
  )

  client.predict(
    practical_subtasks="16 Personality Types of MBTI",
    dropdown_candidates=None,
    api_name="/event_handler_dropdown_candidates",
    
  )
  # print(passableData)
  mbti_prediction=client.predict(
    
    language="English",
    type_modes="Files",
    files=[handle_file(url)],
    video={"video":handle_file(url),"subtitles":None},
    practical_subtasks="16 Personality Types of MBTI",
    pt_scores={"data":passableData,"headers":["Person ID","Path","Openness","Conscientiousness","Extraversion","Agreeableness","Non-Neuroticism"],"metadata":{"display_value":None,"styling":None}},
    dropdown_mbti="The Inspector (ISTJ): Accountant, Auditor, Budget Analyst, Financial Manager, Developer, Systems Analyst, Librarian etc.",
    threshold_mbti=0.5,
    threshold_professional_skills=0.5,
    dropdown_professional_skills=None,
    target_score_ope=0.5,
    target_score_con=0.5,
    target_score_ext=0.5,
    target_score_agr=0.5,
    target_score_nneu=0.5,
    equal_coefficient=0.5,
    number_priority=0.5,
    number_importance_traits=0.5,
    threshold_consumer_preferences=0.5,
    number_openness=0.5,
    number_conscientiousness=0.5,
    number_extraversion=0.5,
    number_agreeableness=0.5,
    number_non_neuroticism=0.5,
    api_name="/event_handler_calculate_practical_task_blocks"
  )
  for entry in mbti_prediction:
    if 'value' in entry and 'data' in entry['value']:
        # Extract the personality type HTML
        personality_type_html = entry['value']['data'][0][2]

        # Use a regex to extract the letters inside span tags
        letters = re.findall(r'>([A-Z])<', personality_type_html)

        # Join the letters into a string
        personality_type = ''.join(letters)
        personality_score = entry['value']['data'][0][3]
        break
  output["MBTI"] = {"type": personality_type, "score": personality_score}
  
  client.predict(
    language="English",
    type_modes="Files",
    practical_tasks="Ranking potential candidates by professional responsibilities",
    practical_subtasks="Professional groups",
    practical_subtasks_selected={"Ranking potential candidates by professional responsibilities":"16 Personality Types of MBTI","Forming effective work teams":"Finding a suitable junior colleague","Predicting consumer preferences for industrial goods":"Car characteristics","Ранжирование потенциальных кандидатов по профессиональным обязанностям":"16 персональных типов личности MBTI","Формирование эффективных рабочих коллективов":"Поиск подходящего младшего коллеги","Прогнозирование потребительских предпочтений в отношении промышленных товаров":"Характеристики автомобиля"},
    api_name="/event_handler_practical_subtasks"
  )

  client.predict(
    practical_subtasks="Professional groups",
    dropdown_candidates="Managers/executives",
    api_name="/event_handler_dropdown_candidates"
  )

  managers_output=client.predict(
    language="English",
    type_modes="Files",
    files=[handle_file(url)],
    video={"video":handle_file(url),"subtitles":None},
    practical_subtasks="Professional groups",
    pt_scores={"data":passableData,"headers":["Person ID","Path","Openness","Conscientiousness","Extraversion","Agreeableness","Non-Neuroticism"],"metadata":{"display_value":None,"styling":None}},
    dropdown_mbti=None,
    threshold_mbti=0.5,
    threshold_professional_skills=0.5,
    dropdown_professional_skills=None,
    target_score_ope=0.5,
    target_score_con=0.5,
    target_score_ext=0.5,
    target_score_agr=0.5,
    target_score_nneu=0.5,
    equal_coefficient=0.5,
    number_priority=0.5,
    number_importance_traits=0.5,
    threshold_consumer_preferences=0.5,
    number_openness=15,
    number_conscientiousness=35,
    number_extraversion=15,
    number_agreeableness=30,
    number_non_neuroticism=5,
    api_name="/event_handler_calculate_practical_task_blocks"
  )
  for entry in managers_output:
    if 'value' in entry and 'data' in entry['value']:
        # Extract the candidate score
        candidate_score = entry['value']['data'][0][2]
        break
  
  output["ProfessionalResponsibilitiesScores"] = {"Managers/executives": candidate_score}
    
  client.predict(
  practical_subtasks="Professional groups",
  dropdown_candidates="Entrepreneurship",
  api_name="/event_handler_dropdown_candidates"
)

  entre_output=client.predict(
    language="English",
    type_modes="Files",
    files=[handle_file(url)],
    video={"video":handle_file(url),"subtitles":None},
    practical_subtasks="Professional groups",
    pt_scores={"data":passableData,"headers":["Person ID","Path","Openness","Conscientiousness","Extraversion","Agreeableness","Non-Neuroticism"],"metadata":{"display_value":None,"styling":None}},
    dropdown_mbti=None,
    threshold_mbti=0.5,
    threshold_professional_skills=0.5,
    dropdown_professional_skills=None,
    target_score_ope=0.5,
    target_score_con=0.5,
    target_score_ext=0.5,
    target_score_agr=0.5,
    target_score_nneu=0.5,
    equal_coefficient=0.5,
    number_priority=0.5,
    number_importance_traits=0.5,
    threshold_consumer_preferences=0.5,
    number_openness=30,
    number_conscientiousness=30,
    number_extraversion=5,
    number_agreeableness=5,
    number_non_neuroticism=30,
    api_name="/event_handler_calculate_practical_task_blocks"
  )
  for entry in entre_output:
    if 'value' in entry and 'data' in entry['value']:
        # Extract the candidate score
        candidate_score = entry['value']['data'][0][2]
        break
  output["ProfessionalResponsibilitiesScores"]["Entrepreneurship"] = candidate_score

  client.predict(
    practical_subtasks="Professional groups",
    dropdown_candidates="Social/Non profit making professions",
    api_name="/event_handler_dropdown_candidates"
  )

  nonprofit_output=client.predict(
    language="English",
    type_modes="Files",
    files=[handle_file(url)],
    video={"video":handle_file(url),"subtitles":None},
    practical_subtasks="Professional groups",
    pt_scores={"data":passableData,"headers":["Person ID","Path","Openness","Conscientiousness","Extraversion","Agreeableness","Non-Neuroticism"],"metadata":{"display_value":None,"styling":None}},
    dropdown_mbti=None,
    threshold_mbti=0.5,
    threshold_professional_skills=0.5,
    dropdown_professional_skills=None,
    target_score_ope=0.5,
    target_score_con=0.5,
    target_score_ext=0.5,
    target_score_agr=0.5,
    target_score_nneu=0.5,
    equal_coefficient=0.5,
    number_priority=0.5,
    number_importance_traits=0.5,
    threshold_consumer_preferences=0.5,
    number_openness=5,
    number_conscientiousness=5,
    number_extraversion=35,
    number_agreeableness=35,
    number_non_neuroticism=20,
    api_name="/event_handler_calculate_practical_task_blocks"
  )
  for entry in nonprofit_output:
    if 'value' in entry and 'data' in entry['value']:
        # Extract the candidate score
        candidate_score = entry['value']['data'][0][2]
        break
  output["ProfessionalResponsibilitiesScores"]["Social/Non profit making professions"] = candidate_score

  client.predict(
    practical_subtasks="Professional groups",
    dropdown_candidates="Public sector professions",
    api_name="/event_handler_dropdown_candidates"
  )

  public_output=client.predict(
    language="English",
    type_modes="Files",
    files=[handle_file(url)],
    video={"video":handle_file(url),"subtitles":None},
    practical_subtasks="Professional groups",
    pt_scores={"data":passableData,"headers":["Person ID","Path","Openness","Conscientiousness","Extraversion","Agreeableness","Non-Neuroticism"],"metadata":{"display_value":None,"styling":None}},
    dropdown_mbti=None,
    threshold_mbti=0.5,
    threshold_professional_skills=0.5,
    dropdown_professional_skills=None,
    target_score_ope=0.5,
    target_score_con=0.5,
    target_score_ext=0.5,
    target_score_agr=0.5,
    target_score_nneu=0.5,
    equal_coefficient=0.5,
    number_priority=0.5,
    number_importance_traits=0.5,
    threshold_consumer_preferences=0.5,
    number_openness=15,
    number_conscientiousness=50,
    number_extraversion=15,
    number_agreeableness=15,
    number_non_neuroticism=5,
    api_name="/event_handler_calculate_practical_task_blocks"
  )
  for entry in public_output:
    if 'value' in entry and 'data' in entry['value']:
        # Extract the candidate score
        candidate_score = entry['value']['data'][0][2]
        break
      
  output["ProfessionalResponsibilitiesScores"]["Public sector professions"] = candidate_score

  client.predict(
    practical_subtasks="Professional groups",
    dropdown_candidates="Scientists/researchers, and engineers",
    api_name="/event_handler_dropdown_candidates"
  )

  science_output=client.predict(
    language="English",
    type_modes="Files",
    files=[handle_file(url)],
    video={"video":handle_file(url),"subtitles":None},
    practical_subtasks="Professional groups",
    pt_scores={"data":passableData,"headers":["Person ID","Path","Openness","Conscientiousness","Extraversion","Agreeableness","Non-Neuroticism"],"metadata":{"display_value":None,"styling":None}},
    dropdown_mbti=None,
    threshold_mbti=0.5,
    threshold_professional_skills=0.5,
    dropdown_professional_skills=None,
    target_score_ope=0.5,
    target_score_con=0.5,
    target_score_ext=0.5,
    target_score_agr=0.5,
    target_score_nneu=0.5,
    equal_coefficient=0.5,
    number_priority=0.5,
    number_importance_traits=0.5,
    threshold_consumer_preferences=0.5,
    number_openness=50,
    number_conscientiousness=15,
    number_extraversion=5,
    number_agreeableness=15,
    number_non_neuroticism=15,
    api_name="/event_handler_calculate_practical_task_blocks"
  )
  for entry in science_output:
    if 'value' in entry and 'data' in entry['value']:
        # Extract the candidate score
        candidate_score = entry['value']['data'][0][2]
        break
  output["ProfessionalResponsibilitiesScores"]["Scientists/researchers, and engineers"] = candidate_score
  # print(data)
  
  return f"Here are the Big 5 Personality data, MBTI personality type & personality score, and Professional Responsibility Scores",output


if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")



# create main function

def main(video_url:str,cv_url:str,prompt:str):
  model = ChatOpenAI(model="gpt-4o")

  prompt_template = ChatPromptTemplate([
      ("system", "You are a helpful AI assistant helping the user evaluate an applicants Big 5 Personality, MBTI personality type and score, professional responsibility scores for job types of Manager/executive, entrepreneurship, social/non profit professions, public sector professions, scientist/research or engineer professions, from a video and summary, and evaluate summary, strengths, and weaknesses from an uploaded PDF CV and evaluate if they are good for a particular job."),
      ( "{prompt}\n  big5personalityTestVideo:{video_url}, cvPdf:{cv_url}"),
  ])
  # prompt_template.format_messages()
  tools = [ predictBig5PersonalityMBTIandProfessionalResponsibilityFromVideo,getSummaryStrengthsWeaknessesFromCV]
  llm_with_tools = model.bind_tools(tools,tool_choice="required")

  message=prompt_template.format_messages(prompt=prompt,video_url=video_url,cv_url=cv_url)

  ai_msg=llm_with_tools.invoke(message)
  message.append(ai_msg)
  for tool_call in ai_msg.tool_calls:
      selected_tool = { "predictBig5PersonalityMBTIandProfessionalResponsibilityFromVideo": predictBig5PersonalityMBTIandProfessionalResponsibilityFromVideo, "getSummaryStrengthsWeaknessesFromCV":getSummaryStrengthsWeaknessesFromCV}[tool_call["name"]]
      tool_msg = selected_tool.invoke(tool_call)
      message.append(tool_msg)

  # print(message)
  output=model.invoke(message)
  print(output.content)

main(prompt="Evaluate a good job for this person",video_url="https://github.com/Vizorous/chamu-videos/raw/refs/heads/main/trim14low.mp4",cv_url="https://raw.githubusercontent.com/Vizorous/chamu-videos/5ceb0aa31831f16c936934f56aa35747082ff6bd/Anushka%20Chandrasena%20-%20CV.pdf")