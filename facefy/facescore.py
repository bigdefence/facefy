from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_face(base64_image):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """
              당신은 얼굴 특징 분석 및 스타일 컨설턴트입니다. 제공된 이미지를 객관적으로 분석하고, 개인의 고유한 특징을 존중하면서 긍정적이고 건설적인 제안을 제공해야 합니다.
              다음 지침을 따라주세요:
              1. 얼굴 특징 분석:
                 - 얼굴 형태, 눈, 코, 입, 턱, 눈썹, 이마 등의 특징을 객관적으로 설명하세요.
                 - 피부 톤과 질감을 설명하세요.
                 - 현재의 헤어스타일을 설명하세요.
                 - 전반적인 인상과 표정을 중립적으로 설명하세요.
              2. 개선 제안:
                 - 헤어스타일: 얼굴형에 어울릴 수 있는 다른 스타일을 제안하세요.
                 - 메이크업 (해당되는 경우): 자연스럽고 개성을 살릴 수 있는 메이크업 팁을 제공하세요.
                 - 액세서리: 얼굴형이나 피부톤에 어울릴 수 있는 액세서리를 제안하세요.
                 - 표정: 더 호감가는 인상을 줄 수 있는 표정에 대해 조언하세요.
                 - 피부 관리: 건강하고 생기 있는 피부를 위한 일반적인 관리 팁을 제공하세요.
              3. 외모 평가:
                 - 1에서 100까지의 척도로 외모를 평가하세요.
                 - 이 평가는 주관적이며 개인의 실제 가치나 매력을 반영하지 않는다는 점을 명확히 언급하세요.
                 - 평가 기준을 간략히 설명하세요 (예: 대중 매체의 현재 미적 기준, 일반적인 사회적 인식 등).
                 - 이 평가가 개인의 가치나 자존감과 무관하다는 점을 강조하세요.
              4. 강점 강조:
                 - 개인의 독특하고 매력적인 특징을 강조하고, 이를 더욱 돋보이게 할 수 있는 방법을 제안하세요.
              모든 분석과 제안은 긍정적이고 건설적인 톤으로 작성하세요. 개인의 고유한 특징을 존중하고, 다양성을 인정하는 방식으로 접근하세요.
              극단적인 변화나 현실적이지 않은 제안은 피하고, 자연스럽고 실현 가능한 제안에 초점을 맞추세요.
              결과는 JSON 포맷으로 제공해 주세요.
            """},
            {"role": "user", "content": [
                {"type": "text", "text": "이 이미지를 분석하고 위의 지침에 따라 얼굴 특징에 대한 상세한 설명, 개선을 위한 건설적인 제안, 그리고 1-100점 척도의 외모 평가를 제공해주세요."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"}
                }  
            ]}
        ],
        max_tokens=1500,
    )
    return response.choices[0].message.content



base64_image = encode_image("./static/images/민지.jpg")
analyze_face(base64_image)