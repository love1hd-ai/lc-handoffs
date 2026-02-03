"""
Handoffs Agent: 基于 LangChain Handoffs 模式的实现
使用本地模型解释图片，然后使用远程模型提取有用内容

参考: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
"""

import os
import json
import re
from pathlib import Path
from typing import Optional, Literal, List, Dict, Any
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ValidationError
import ollama
from ollama import Image
from handoffs.client_all import client_local, client_remote

# 默认模型配置
MODEL_REMOTE = "deepseek-v3.1:671b-cloud"
MODEL_LOCAL = "deepseek-ocr"


class ProcessingStep(str, Enum):
    """处理步骤枚举"""

    INITIAL = "initial"
    IMAGE_ANALYSIS = "image_analysis"
    EXTRACTION = "extraction"
    COMPLETED = "completed"
    ERROR = "error"


class Consumption(BaseModel):
    """消费结构体"""

    date: str = Field(..., description="消费时间, 格式: yyyy-mm-dd hh:mm:ss")
    item: str = Field(..., description="消费内容/项目")
    category: str = Field(..., description="消费类型/分类")
    amount: Decimal = Field(..., description="消费金额(负数表示支出)")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """验证日期时间格式"""
        try:
            # 尝试解析 yyyy-mm-dd hh:mm:ss 格式
            datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
            return v
        except ValueError:
            try:
                # 也支持 yyyy-mm-dd 格式
                date.fromisoformat(v)
                return v
            except ValueError:
                raise ValueError(
                    f"日期格式错误, 应为 yyyy-mm-dd hh:mm:ss 或 yyyy-mm-dd, 收到: {v}"
                )

    @field_validator("amount", mode="before")
    @classmethod
    def validate_amount(cls, v) -> Decimal:
        """验证金额并转换为 Decimal(允许负数表示支出)"""
        if isinstance(v, (int, float, str)):
            decimal_value = Decimal(str(v))
        elif isinstance(v, Decimal):
            decimal_value = v
        else:
            raise ValueError(f"金额类型错误, 应为数字或字符串, 收到: {type(v)}")

        if decimal_value == 0:
            raise ValueError(f"消费金额不能为0，收到: {decimal_value}")
        return decimal_value

    def __str__(self) -> str:
        return f"消费记录: {self.date} | [{self.category}] {self.item} | ¥{self.amount:.2f}"


class HandoffsState(BaseModel):
    """
    Handoffs 状态管理
    基于 LangChain Handoffs 模式: 状态驱动行为, 工具更新状态触发转换
    """

    current_step: ProcessingStep = Field(
        default=ProcessingStep.INITIAL, description="当前处理步骤"
    )
    image_path: Optional[str] = Field(default=None, description="图片路径")
    input_text: Optional[str] = Field(default=None, description="输入文本")
    local_description: Optional[str] = Field(
        default=None, description="本地模型分析的图片描述"
    )
    extracted_content: Optional[Any] = Field(
        default=None, description="远程模型提取的内容"
    )
    consumptions: List[Dict[str, Any]] = Field(
        default_factory=list, description="解析后的消费记录列表"
    )
    error: Optional[str] = Field(default=None, description="错误信息")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="元数据(用于存储额外信息)"
    )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于 JSON 序列化"""
        result = self.model_dump(mode="json")
        # 确保 consumptions 是列表格式
        if self.extracted_content:
            if isinstance(self.extracted_content, Consumption):
                result["consumption"] = self.extracted_content.model_dump(mode="json")
            elif isinstance(self.extracted_content, list) and all(
                isinstance(c, Consumption) for c in self.extracted_content
            ):
                result["consumptions"] = [
                    c.model_dump(mode="json") for c in self.extracted_content
                ]
        return result


class HandoffsAgent:
    """
    Handoffs Agent - 基于 LangChain Handoffs 模式实现
    核心机制: 工具更新状态变量, 系统根据状态调整行为
    参考: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
    """

    def __init__(
        self,
        model_local: str = MODEL_LOCAL,
        client_local: Optional[ollama.Client] = None,
        model_remote: str = MODEL_REMOTE,
        client_remote: Optional[ollama.Client] = None,
    ):
        """
        初始化 Handoffs Agent

        Args:
            model_local: 本地 ollama 模型名称(默认: deepseek-ocr)
            client_local: 本地 ollama 客户端(默认使用 client_all.client_local)
            model_remote: 远程 ollama 模型名称(默认: deepseek-v3.1:671b-cloud)
            client_remote: 远程 ollama 客户端(默认使用 client_all.client_remote)
        """
        self.model_local = model_local
        self.model_remote = model_remote
        # 使用传入的客户端或默认客户端
        self.client_local = (
            client_local if client_local is not None else globals()["client_local"]
        )
        self.client_remote = (
            client_remote if client_remote is not None else globals()["client_remote"]
        )

    def analyze_image(self, state: HandoffsState) -> HandoffsState:
        """
        工具函数：分析图片并更新状态
        这是 handoffs 模式中的工具，通过更新状态触发转换

        Args:
            state: 当前状态

        Returns:
            HandoffsState: 更新后的状态
        """
        if not state.image_path:
            state.current_step = ProcessingStep.ERROR
            state.error = "图片路径未提供"
            return state

        if not Path(state.image_path).exists():
            state.current_step = ProcessingStep.ERROR
            state.error = f"图片文件不存在: {state.image_path}"
            return state

        print(f"[本地模型] 正在使用 {self.model_local} 分析图片...")
        try:
            response = self.client_local.chat(
                model=self.model_local,
                messages=[
                    {
                        "role": "user",
                        "content": state.metadata.get("image_prompt", "提取图片内容"),
                        "images": [Image(value=state.image_path)],
                    }
                ],
            )
            description = response["message"]["content"]
            print(f"[本地模型] 分析完成")

            # 更新状态：工具更新状态变量触发转换
            state.local_description = description
            state.current_step = ProcessingStep.EXTRACTION
            return state
        except Exception as e:
            state.current_step = ProcessingStep.ERROR
            state.error = f"本地模型分析失败: {str(e)}"
            return state

    def extract_content(self, state: HandoffsState) -> HandoffsState:
        """
        工具函数：提取有用内容并更新状态
        这是 handoffs 模式中的工具，通过更新状态触发转换

        Args:
            state: 当前状态

        Returns:
            HandoffsState: 更新后的状态
        """
        # 确定要提取的文本
        text_to_extract = state.input_text or state.local_description
        if not text_to_extract:
            state.current_step = ProcessingStep.ERROR
            state.error = "没有可提取的文本内容"
            return state

        extraction_prompt = state.metadata.get("extraction_prompt")
        if extraction_prompt is None:
            from datetime import datetime

            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            extraction_prompt_template = """请从以下文本中提取消费信息, 包括:
1. 消费时间, 格式: yyyy-mm-dd hh:mm:ss(如果文本中没有明确时间, 使用当前时间)
2. 消费内容/项目(如果文本中有地点信息, 格式为"项目(地点)", 如"在大丰买衣服"->"衣服(大丰)"; 如果原文本已包含括号, 保持原格式)
3. 消费类型/分类(格式: 具体分类-动作, 如"服装-购买"、"餐饮-外卖"等)
4. 消费金额(支持中文数字, 如"四块钱"转换为4.00, "一百元"转换为100.00, "四十五"转换为45.00)

提取规则:
- 如果消费项目包含括号, 例如"书亦烧仙草(爱尚里店)外卖订单":
  * item: "书亦烧仙草(爱尚里店)"(包含括号的完整内容, 到括号结束)
  * category: "餐饮-外卖"(原分类的前部分-括号后的关键词, 去掉"订单"等后缀词)
- 如果消费项目不包含括号, 但有地点信息(如"在XX"、"XX店"、"XX买"等):
  * item: "项目(地点)"(将地点信息添加到括号中), 例如"在大丰买衣服"->item: "衣服(大丰)"
  * category: "具体分类-动作"(根据消费内容推断具体分类, 如"服装-购买"、"餐饮-外卖"等)
- 如果消费项目不包含括号和地点, 例如"午餐"、"吃饭":
  * item: "午餐" 或 "吃饭"(提取消费项目名称)
  * category: "餐饮"(根据消费内容推断分类, 如: 餐饮、交通、购物等)
- 金额提取:
  * 支持阿拉伯数字: 如"花了100元"、"4块钱"、"¥50"
  * 支持中文数字: 如"四块钱"->4.00, "一百元"->100.00, "五十块"->50.00, "四十五"->45.00
  * 金额可以是正数(收入)或负数(支出), 支出通常为负数

重要: 请积极提取信息, 即使文本比较口语化或简短. 只要文本中包含消费相关的信息(如"花了"、"买了"、"消费"等关键词), 就应该提取出来.

当前时间: {time_now}

示例输出(必须是有效的 JSON 格式数组):
[
    {{
        "date": "2024-01-01 10:00:00",
        "item": "书亦烧仙草(爱尚里店)",
        "category": "餐饮-外卖",
        "amount": 100.00
    }},
    {{
        "date": "2024-01-01 10:00:00",
        "item": "午餐",
        "category": "餐饮",
        "amount": 50.00
    }},
    {{
        "date": "{time_now}",
        "item": "吃饭",
        "category": "餐饮",
        "amount": 4.00
    }},
    {{
        "date": "{time_now}",
        "item": "衣服(大丰)",
        "category": "服装-购买",
        "amount": 45.00
    }}
]

文本内容:
{text}"""
            extraction_prompt = extraction_prompt_template.format(
                time_now=time_now, text=text_to_extract
            )

        print(f"[远程模型] 正在使用 {self.model_remote} 提取有用内容...")
        try:
            response = self.client_remote.chat(
                model=self.model_remote,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的信息提取助手。请只返回有效的 JSON 格式数据。",
                    },
                    {
                        "role": "user",
                        "content": extraction_prompt,
                    },
                ],
            )

            extracted_text = response["message"]["content"]
            print(f"[远程模型] 提取完成")

            # 解析 JSON
            extracted_content = self._parse_extracted_content(extracted_text)

            # 更新状态：工具更新状态变量触发转换
            state.extracted_content = extracted_content
            if isinstance(extracted_content, Consumption):
                state.consumptions = [extracted_content.model_dump(mode="json")]
            elif isinstance(extracted_content, list) and all(
                isinstance(c, Consumption) for c in extracted_content
            ):
                state.consumptions = [
                    c.model_dump(mode="json") for c in extracted_content
                ]
            state.current_step = ProcessingStep.COMPLETED
            return state
        except Exception as e:
            state.current_step = ProcessingStep.ERROR
            state.error = f"远程模型提取失败: {str(e)}"
            return state

    def _parse_extracted_content(self, extracted_text: str) -> Any:
        """
        解析提取的内容（内部辅助方法）

        Args:
            extracted_text: 远程模型返回的文本

        Returns:
            Consumption 对象、列表或原始文本
        """
        try:
            # 清理可能的 markdown 代码块标记
            extracted_text = re.sub(r"```json\s*", "", extracted_text)
            extracted_text = re.sub(r"```\s*$", "", extracted_text)
            extracted_text = extracted_text.strip()

            # 解析 JSON
            extracted_json = json.loads(extracted_text)

            # 验证并转换为 Consumption 对象
            if isinstance(extracted_json, list):
                consumptions = []
                for item in extracted_json:
                    if isinstance(item, dict):
                        try:
                            consumption = Consumption(**item)
                            consumptions.append(consumption)
                        except ValidationError as ve:
                            print(f"[警告] 跳过无效的消费记录: {ve}")
                return consumptions if consumptions else extracted_text
            elif isinstance(extracted_json, dict):
                try:
                    return Consumption(**extracted_json)
                except ValidationError as ve:
                    print(f"[警告] JSON 数据验证失败: {ve}")
                    return extracted_text
            else:
                return extracted_text

        except json.JSONDecodeError as je:
            print(f"[警告] JSON 解析失败: {je}")

            # 尝试从文本中提取 JSON
            start_idx = extracted_text.find("[")
            end_idx = extracted_text.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_candidate = extracted_text[start_idx : end_idx + 1]
                try:
                    extracted_json = json.loads(json_candidate)
                    if isinstance(extracted_json, list):
                        consumptions = []
                        for item in extracted_json:
                            if isinstance(item, dict):
                                try:
                                    consumption = Consumption(**item)
                                    consumptions.append(consumption)
                                except ValidationError:
                                    pass
                        if consumptions:
                            return consumptions
                except (json.JSONDecodeError, ValidationError):
                    pass

            # 尝试单个对象
            start_idx = extracted_text.find("{")
            end_idx = extracted_text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_candidate = extracted_text[start_idx : end_idx + 1]
                try:
                    extracted_json = json.loads(json_candidate)
                    if isinstance(extracted_json, dict):
                        return Consumption(**extracted_json)
                except (json.JSONDecodeError, ValidationError):
                    pass

            return extracted_text

    def route(
        self, state: HandoffsState
    ) -> Literal["analyze_image", "extract_content", "end"]:
        """
        路由函数：根据当前状态决定下一步操作
        这是 handoffs 模式中的路由逻辑

        Args:
            state: 当前状态

        Returns:
            下一步要执行的工具名称或 "end"
        """
        if state.current_step == ProcessingStep.ERROR:
            return "end"

        if state.current_step == ProcessingStep.COMPLETED:
            return "end"

        if state.current_step == ProcessingStep.INITIAL:
            if state.image_path:
                return "analyze_image"
            elif state.input_text:
                return "extract_content"
            else:
                state.current_step = ProcessingStep.ERROR
                state.error = "必须提供 image_path 或 input_text"
                return "end"

        if state.current_step == ProcessingStep.IMAGE_ANALYSIS:
            # 当状态是 IMAGE_ANALYSIS 时，继续执行 analyze_image
            return "analyze_image"

        if state.current_step == ProcessingStep.EXTRACTION:
            # 当状态是 EXTRACTION 时，需要执行 extract_content
            return "extract_content"

        return "end"

    def process(
        self,
        image_path: Optional[str] = None,
        text: Optional[str] = None,
        image_prompt: str = "提取图片内容",
        extraction_prompt: str = None,
    ) -> Dict[str, Any]:
        """
        完整的处理流程：基于 handoffs 模式的状态驱动处理
        参考: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs

        Args:
            image_path: 图片路径（可选）
            text: 文本内容（可选，如果提供则跳过图片分析）
            image_prompt: 图片分析提示词
            extraction_prompt: 信息提取提示词

        Returns:
            dict: 包含处理结果的字典
        """
        # 初始化状态
        state = HandoffsState(
            current_step=ProcessingStep.INITIAL,
            image_path=image_path,
            input_text=text,
            metadata={
                "image_prompt": image_prompt,
                "extraction_prompt": extraction_prompt,
            },
        )

        # Handoffs 模式：根据状态路由并执行工具
        max_iterations = 10  # 防止无限循环
        iteration = 0

        while (
            state.current_step != ProcessingStep.COMPLETED
            and state.current_step != ProcessingStep.ERROR
        ):
            if iteration >= max_iterations:
                state.current_step = ProcessingStep.ERROR
                state.error = "处理超时：达到最大迭代次数"
                break

            # 路由：根据状态决定下一步
            next_action = self.route(state)

            if next_action == "end":
                break
            elif next_action == "analyze_image":
                state = self.analyze_image(state)
            elif next_action == "extract_content":
                state = self.extract_content(state)

            iteration += 1

        # 返回结果
        return state.to_dict()


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Handoffs Agent: 基于 LangChain Handoffs 模式实现"
    )
    parser.add_argument("-i", "--image", type=str, help="图片路径", default=None)
    parser.add_argument("-t", "--text", type=str, help="文本内容", default=None)
    parser.add_argument(
        "--local-model",
        type=str,
        default=MODEL_LOCAL,
        help=f"本地 ollama 模型名称(默认: {MODEL_LOCAL})",
    )
    parser.add_argument(
        "--remote-model",
        type=str,
        default=MODEL_REMOTE,
        help=f"远程 ollama 模型名称(默认: {MODEL_REMOTE})",
    )

    args = parser.parse_args()

    if not args.image and not args.text:
        parser.error("必须提供 -i (图片路径) 或 -t (文本内容)")

    # 创建 agent
    agent = HandoffsAgent(
        model_local=args.local_model,
        model_remote=args.remote_model,
    )

    try:
        # 处理
        result = agent.process(image_path=args.image, text=args.text)

        # 输出结果
        print("=" * 60)
        print("处理结果")
        print("=" * 60)
        print(json.dumps(result, indent=4, ensure_ascii=False))
        print("=" * 60)

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
