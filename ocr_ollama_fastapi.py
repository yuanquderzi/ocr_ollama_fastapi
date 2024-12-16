from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from PIL import Image
import io
import logging
import traceback
import numpy as np
import requests
import json

# 初始化 FastAPI 应用
app = FastAPI()

# 初始化 PaddleOCR 模型
ocr_model = PaddleOCR(use_angle_cls=True, lang="ch")

# 定义ollama模型相关配置
OLLAMA_API_URL = "http://localhost:11433/api/generate"
OLLAMA_MODEL_NAME = "llama3"

@app.post("/extract-content/")
async def extract_content(file: UploadFile):
    try:
        logging.debug("开始处理上传文件")

        # 验证文件类型
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="上传的文件必须是图片格式")
        
        # 读取上传图片
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # 将 PIL.Image 转为 np.ndarray
        image_array = np.array(image)
        logging.debug("图片加载成功")
        
        # OCR 识别
        result = ocr_model.ocr(image_array, cls=True)
        logging.debug(f"OCR 识别结果: {result}")
        extracted_texts = [line[1][0] for res in result for line in res]

        # 指定的关键信息
        key = "姓名和职位"
        
        # 构建 Prompt
        prompt = f"""你的任务是从OCR文字识别的结果中提取指定的关键信息。OCR识别结果用```符号包裹，其中包含按照原始图像从左至右、从上至下顺序排列的文本内容。我指定的关键信息用[]符号包裹。

需要注意的是：
1. OCR识别结果可能包含长句被切断、不规范分词、内容错位等问题。
2. 请结合上下文语义进行综合判断，提取出准确的关键信息。
3. 如果OCR结果中找不到指定的关键信息，请将其值设置为“未找到相关信息”。

请以JSON格式返回结果，其中包含以下key：
- 姓名: OCR结果中的姓名。
- 职位: OCR结果中的职位。

只输出JSON格式的结果，不要包含其他内容！现在开始：

OCR文字：```{extracted_texts}```

指定的关键信息：[姓名, 职位]。
"""
        logging.debug(f"Prompt 构建成功: {prompt}")

        # 调用 Ollama 模型
        headers = {"Content-Type": "application/json"}
        data = {
            "model": OLLAMA_MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, headers=headers, data=json.dumps(data))
        if response.status_code != 200:
            logging.error(f"调用 Ollama 模型失败: {response.text}")
            raise HTTPException(status_code=500, detail="调用大模型失败")

        # 解析模型响应
        model_response = response.json()
        if "response" in model_response:
            try:
                # 尝试解析为 JSON 对象
                extracted_info = json.loads(model_response["response"])
            except json.JSONDecodeError:
                logging.error(f"解析响应为 JSON 时失败: {model_response['response']}")
                raise HTTPException(status_code=500, detail="模型响应格式错误")
        else:
            extracted_info = {"error": "模型未返回有效数据"}
        
        # 返回结果
        return JSONResponse(content=extracted_info)
    
    except Exception as e:
        # 打印详细错误堆栈
        error_message = traceback.format_exc()
        logging.error(f"发生错误:\n{error_message}")
        raise HTTPException(status_code=500, detail="内部服务器错误")

