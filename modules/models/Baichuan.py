from typing import List, Generator
import httpx
from ..utils import *

from .base_model import BaseLLMModel

from ..index_func import *

import logging
import httpx
from typing import List, Iterator

logger = logging.getLogger(__name__)

CONFIG = {
  "Baichuan": "http://192.168.1.181:8008/api/v1/nlp/baichuan",
  "timeout": 120,
  "token": "611cf2c6-5fd9-11ed-a9fa-88e9fe79f123",
  "history_remeber_number": 10
}


class Post_Client():
    """基础大模型"""

    def __init__(self, conf: dict) -> None:
        self.conf = conf
        self.headers = {"Authorization": conf["token"]}
        self.timeout = conf["timeout"]

    def _call(self, llm: str, messages: List[dict]) -> str:
        """执行调用"""

        url = self.conf[llm]
        query = {"messages": messages, "stream": False}
        logger.info("[llm call] url: <{}>, messages: <{}>".format(url, query))
        response = httpx.post(url, json=query, headers=self.headers, timeout=self.timeout)

        if response.status_code == 200 and response.json()["header"]["code"] == 200:
            content = response.json()["body"]["content"]
            logger.info("[llm call] response content: <{}>".format(content))
            return content
        logger.error(response)

    def _stream_call(self, llm: str, messages: List[dict]) -> Iterator[str]:
        """流式调用"""

        url = self.conf[llm]
        query = {"messages": messages, "stream": True}

        with httpx.stream("POST", url=url, json=query, headers=self.headers, timeout=self.timeout,) as response:
            lines = ''
            for line in response.iter_lines():
                pattern = re.compile(r'"content": "(.*?)"')
                matches = pattern.findall(line)
                if matches:
                    line = matches[0]
                    line=line.replace("\\n", "\n")
                    lines += line
                    yield lines

class Baichuan_Client(BaseLLMModel):
    def __init__(self, model_name, user_name="") -> None:
        super().__init__(model_name=model_name, user=user_name)
        self.llm_client = Post_Client(conf=CONFIG)
        self.llm = "Baichuan"  # Correct the llm value to match the key in the CONFIG dictionary
        self.history_remeber_number = CONFIG["history_remeber_number"]

    def _get_baichuan_style_input(self):
        history = [x["content"] for x in self.history]
        query = history.pop()
        history = history[-self.history_remeber_number:]
        history = "\n".join(f"user: {history[i]}\nsystem: {history[i+1]}" for i in range(0, len(history), 2))
        # history = [{"role": "assistant", "content": history[i]} for i in range(0, len(history), 2)]
        return history, query

    def _get_baichuan_message(self, query, history):
        return [{"role": "user", "content": history + "\nuser:" + query + "\nsystem:"}]

    def get_answer_at_once(self):
        history, query = self._get_baichuan_style_input()
        message = self._get_baichuan_message(query, "")
        
        response = self.llm_client._call(self.llm, messages=message)  # Use the _call method for non-streaming
        return response, len(response)

    def get_answer_stream_iter(self) -> Generator[str, None, None]:
        history, query = self._get_baichuan_style_input()
        message = self._get_baichuan_message(query, history)

        for chunk in self.llm_client._stream_call(self.llm, messages=message):  # Use the _stream_call method for streaming
            yield chunk

    def summarize_index(self, files, chatbot, language):
        status = gr.Markdown.update()
        if files:
            index = construct_index(self.api_key, file_src=files)
            status = i18n("总结完成")
            logging.info(i18n("生成内容总结中……"))

            prompt_template = (
                "根据下面的内容用"
                + language
                + "简洁地写一段总结:"
            )

            result_string = ''.join([doc.page_content for doc in list(index.docstore.__dict__["_dict"].values())])
            prompt = prompt_template + result_string
            message = self._get_baichuan_message(prompt, "")
            summary = self.llm_client._call(self.llm, messages=message)
            chatbot.append([i18n("上传了") + str(len(files)) + "个文件", summary])
        return chatbot, status
