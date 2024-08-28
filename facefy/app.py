from flask import Flask, request, render_template, jsonify,session,redirect
from io import BytesIO
import base64
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from PIL import Image
import google.generativeai as genai
import hashlib
# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
gemini_key=os.getenv('GEMINI_API_KEY')
# Initialize OpenAI client
client = OpenAI(api_key=api_key)
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-1.5-flash')
app = Flask(__name__)
app.secret_key = 'face_bigdefence'  # Set a proper secret key for your session

def encode_image(image_file):
    """ Encode the image file in base64. """
    # Use BytesIO to handle image data
    buffered = BytesIO()
    image_file.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_face(base64_image):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                  당신은 전문적인 이미지 및 스타일 컨설턴트입니다. 제공된 이미지를 객관적으로 분석하시고, 개인의 고유한 특징을 존중하면서 긍정적이고 건설적인 제안을 제공해 주셔야 합니다. 다음 지침에 따라 분석을 수행하시고, 결과를 지정된 JSON 형식으로 반환해 주시기 바랍니다:

                        {
                          "face_feature_analysis": {
                            "face_shape": "얼굴형 설명",
                            "eyes": "눈 설명",
                            "nose": "코 설명",
                            "mouth": "입 설명",
                            "chin": "턱 설명",
                            "eyebrows": "눈썹 설명",
                            "forehead": "이마 설명",
                            "skin_tone": "피부톤 설명",
                            "skin_texture": "피부 질감 설명",
                            "current_hairstyle": "현재 헤어스타일 설명",
                            "overall_impression": "전반적인 인상 설명"
                          },
                          "improvement_suggestions": {
                            "hairstyle": "헤어스타일 제안",
                            "makeup": "메이크업 팁 (해당되는 경우)",
                            "accessories": "액세서리 제안",
                            "expression": "표정 조언",
                            "skin_care": "피부 관리 팁"
                          },
                          "appearance_evaluation": {
                            "score": "1-100 사이의 점수",
                            "explanation": "평가 기준 설명 및 주관성/가치 무관성 강조"
                          },
                          "highlight_strengths": {
                            "unique_features": "독특하고 매력적인 특징 나열",
                            "enhancement_suggestions": "이러한 특징을 더욱 돋보이게 할 방법"
                          }
                        }

                        모든 분석과 제안은 긍정적이고 건설적인 톤으로 작성해 주시기 바랍니다. 극단적인 변화나 비현실적인 제안은 피하시고, 자연스럽고 실현 가능한 제안에 초점을 맞추어 주시기 바랍니다. 결과는 반드시 위에 지정된 JSON 형식으로 제공해 주시며, 모든 필드를 포함해 주셔야 합니다. 항상 한글로 출력해 주십시오.
                """},
                {"role": "user", "content": [
                    {"type": "text", "text": "Please analyze this image according to the above guidelines, providing a detailed description of facial features, constructive improvement suggestions, and an appearance evaluation on a 1-100 scale."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"}
                    }  
                ]}
            ],
            max_tokens=1500,
        )
        content = response.choices[0].message.content.strip()
        
        # Clean and prepare the content for JSON parsing
        if content.startswith('```json'):
            content = content[8:].strip()  # Remove the starting '```json'
        if content.endswith('```'):
            content = content[:-3].strip()  # Remove the ending '```'
        
        # Attempt to parse the JSON content
        try:
            parsed_data = json.loads(content)
        except json.JSONDecodeError as e:
            return {"error": "Failed to parse JSON", "raw_content": content, "exception": str(e)}
        
        # Standardize the keys in the response
        standardized_response = {}

        # Map different possible keys to the standard key
        standardized_response['face_feature_analysis'] = parsed_data.get('face_feature_analysis') or parsed_data.get('facial_feature_analysis', {})
        standardized_response['improvement_suggestions'] = parsed_data.get('improvement_suggestions', {})
        standardized_response['appearance_evaluation'] = parsed_data.get('appearance_evaluation', {})
        standardized_response['highlight_strengths'] = parsed_data.get('highlight_strengths', {})

        # Return the standardized response
        return standardized_response
    except Exception as e:
        # Handle any exceptions and provide a clear error message
        return {"error": f"Error processing image: {str(e)}"}
def physiognomy(content):
    prompt = f"""
    당신은 관상학의 전문가로서, 다음 얼굴 특징을 바탕으로 깊이 있는 관상학적 분석을 제공해 주세요:

    - **얼굴형**: {content['face_feature_analysis']['face_shape']}
    - **이마**: {content['face_feature_analysis']['forehead']}
    - **눈**: {content['face_feature_analysis']['eyes']}
    - **코**: {content['face_feature_analysis']['nose']}
    - **입**: {content['face_feature_analysis']['mouth']}

    제공해주셔야 할 내용은 다음과 같습니다:

    1. **얼굴 특징 분석**:
       - 각 얼굴 특징이 관상학적으로 어떤 의미를 가지는지 설명해 주세요. (예: 얼굴형, 이마의 크기와 형태, 눈의 위치와 모양, 코의 형태, 입의 크기와 위치 등)
       - 이 분석을 통해 개별 얼굴 특징이 서로 어떻게 연관되어 있는지 설명해 주세요.

    2. **관상학적 분석**:
       - **성격 특성**: 각 얼굴 특징이 개인의 성격, 성향, 대인 관계 스타일에 대해 무엇을 시사하는지 분석해 주세요. (예: 이마가 넓은 사람은 지혜롭고 계획적일 가능성이 높다는 식으로)
       - **건강 통찰**: 얼굴 특징으로부터 유추할 수 있는 건강 관련 정보나 경고 신호를 제공해 주세요. (예: 얼굴의 특정 부분이 두드러지면 특정 건강 문제의 가능성이 있는지)
       - **운세와 전망**: 얼굴 특징이 개인의 운세, 미래 전망, 인생의 방향성에 어떤 영향을 미칠 수 있는지 설명해 주세요. (예: 특정 얼굴형이 직업적 성공이나 개인적 행복과 어떤 관계가 있는지)

    분석이 상세하고 통찰력 있는 정보를 포함하도록 해 주세요. 얼굴 특징 각각의 중요성과 이들이 개인의 전반적인 성격과 운세에 미치는 영향을 잘 설명해 주시기 바랍니다.
    응답은 다음 JSON 형식으로 구성해 주세요:
    {{
      "facial_feature_analysis": {{
        "face_shape": "얼굴형 분석 결과",
        "forehead": "이마 분석 결과",
        "eyes": "눈 분석 결과",
        "nose": "코 분석 결과",
        "mouth": "입 분석 결과"
      }},
      "physiognomy_analysis": {{
        "personality_traits": "성격 특성 분석 결과",
        "health_insights": "건강 통찰 결과",
        "fortune_prospects": "운세와 전망 분석 결과"
      }}
    }}
    """
    content = model.generate_content(prompt)
    content = content.text
    if content.startswith('```json'):
        content = content[8:].strip()  # Remove the starting '```json'
    if content.endswith('```'):
        content = content[:-3].strip()  # Remove the ending '```'
    try:
        parsed_data = json.loads(content)
    except json.JSONDecodeError as e:
        return {"error": "Failed to parse JSON", "raw_content": content, "exception": str(e)}
    
    return parsed_data


def fortune_tell(content):
    prompt = f"""
    당신은 관상학 및 운세 분석의 전문가입니다. 다음 얼굴 특징을 바탕으로 개인의 다양한 운세를 전문적으로 분석해 주세요:

    - **얼굴형**: {content['face_feature_analysis']['face_shape']}
    - **이마**: {content['face_feature_analysis']['forehead']}
    - **눈**: {content['face_feature_analysis']['eyes']}
    - **코**: {content['face_feature_analysis']['nose']}
    - **입**: {content['face_feature_analysis']['mouth']}

    아래의 운세를 전문적으로 분석해 주세요:

    1. **연애운**:
       - **성격과 연애**: 얼굴 특징이 개인의 연애 성향, 대인 관계 스타일에 어떻게 영향을 미치는지 설명해 주세요.
       - **연애 전망**: 현재와 미래의 연애운에 대해 설명해 주세요.

    2. **금전운**:
       - **재정 성향**: 얼굴 특징이 개인의 금전 관리 스타일, 재정적 태도와 어떻게 연관되어 있는지 분석해 주세요.
       - **금전 운세**: 현재와 미래의 금전 운에 대해 설명해 주세요.

    3. **건강운**:
       - **건강 상태**: 얼굴 특징이 건강 상태와 어떻게 연관되어 있는지 설명해 주세요.
       - **건강 전망**: 현재와 미래의 건강 운세에 대해 설명해 주세요.

    4. **성공운**:
       - **직업과 성공**: 얼굴 특징이 개인의 직업적 성공, 경력 개발과 어떻게 연관되어 있는지 분석해 주세요.
       - **성공 전망**: 현재와 미래의 성공 운세에 대해 설명해 주세요.

    5. **기타 운세**:
       - **총괄 운세**: 얼굴 특징을 바탕으로 개인의 전체적인 운세를 종합적으로 분석해 주세요.

    분석이 상세하고 통찰력 있는 정보를 포함하도록 해 주세요. 얼굴 특징 각각의 중요성과 이들이 개인의 다양한 운세에 미치는 영향을 잘 설명해 주시기 바랍니다.
    응답은 다음 JSON 형식으로 구성해 주세요:

    {{
      "fortune_analysis": {{
        "love_life": {{
          "personality_and_relationships": "연애 성향 및 대인 관계 분석 결과",
          "love_prospects": "연애 전망 분석 결과"
        }},
        "financial_status": {{
          "financial_traits": "재정 성향 분석 결과",
          "financial_prospects": "금전 운세 분석 결과"
        }},
        "health_status": {{
          "health_state": "건강 상태 분석 결과",
          "health_prospects": "건강 전망 분석 결과"
        }},
        "success_prospects": {{
          "career_success": "직업적 성공 분석 결과",
          "success_prospects": "성공 전망 분석 결과"
        }},
        "overall_fortune": "총괄 운세 분석 결과"
      }}
    }}
    """
    content = model.generate_content(prompt)
    content = content.text
    if content.startswith('```json'):
        content = content[8:].strip()  # Remove the starting '```json'
    if content.endswith('```'):
        content = content[:-3].strip()  # Remove the ending '```'
    try:
        parsed_data = json.loads(content)
    except json.JSONDecodeError as e:
        return {"error": "Failed to parse JSON", "raw_content": content, "exception": str(e)}
    
    return parsed_data

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/facial-analysis')
def facial_analysis():
    # Check if analysis response exists in the session
    if 'analysis_response' not in session:
        return redirect('/')  # Redirect to the main page if no analysis data is present

    # Check if physiognomy results are already cached in the session
    if 'physiognomy_result' in session:
        physiognomy_result = session['physiognomy_result']
    else:
        # Generate the physiognomy results and store them in the session
        physiognomy_result = physiognomy(session['analysis_response'])
        session['physiognomy_result'] = physiognomy_result

    return render_template('facial-analysis.html', physiognomy=physiognomy_result)

@app.route('/fortune-telling')
def fortune_telling():
    # Check if analysis response exists in the session
    if 'analysis_response' not in session:
        return redirect('/')  # Redirect to the main page if no analysis data is present

    # Check if fortune-telling results are already cached in the session
    if 'fortune_result' in session:
        fortune_result = session['fortune_result']
    else:
        # Generate the fortune-telling results and store them in the session
        fortune_result = fortune_tell(session['analysis_response'])
        session['fortune_result'] = fortune_result

    return render_template('fortune-telling.html', fortune=fortune_result)




@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image']
    
    try:
        # Create a unique identifier for the uploaded image
        image_content = image.read()
        image_hash = hashlib.md5(image_content).hexdigest()

        # Check if the current image is the same as the one stored in the session
        if session.get('image_hash') == image_hash:
            # If the image is the same, use the existing analysis result
            if 'analysis_response' in session:
                return render_template('result.html', result=session['analysis_response'])
        else:
            # Reset the file pointer to the start of the file
            image.seek(0)
        
        # Encode the image in base64
        base64_image = encode_image(Image.open(BytesIO(image_content)))

        # Analyze the face
        analysis_result = analyze_face(base64_image)

        # Check for errors in the response
        if 'error' in analysis_result:
            return jsonify({'error': analysis_result['error']}), 500

        # Store the new analysis result and image hash in the session
        session['analysis_response'] = analysis_result
        session['image_hash'] = image_hash

        # Render the result in result.html
        return render_template('result.html', result=analysis_result)
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500



if __name__ == '__main__':
    app.run(debug=True)