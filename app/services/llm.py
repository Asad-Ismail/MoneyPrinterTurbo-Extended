import json
import logging
import re
from typing import Any, List

import g4f
import requests
from loguru import logger
from openai import AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion

from app.config import config

_max_retries = 5
OPENAI_COMPATIBLE_PROVIDERS = {"openai", "groq", "openrouter", "moonshot", "deepseek", "oneapi", "ollama"}


def normalize_error(error: Exception | str) -> str:
    return str(error).strip()


def _get_provider_runtime(provider: str | None = None) -> dict[str, Any]:
    provider_name = (provider or config.app.get("llm_provider", "openai")).lower()
    provider_cfg = config.get_llm_provider_config(provider_name)
    provider_cfg["provider"] = provider_name

    if provider_name == "openai":
        provider_cfg["base_url"] = provider_cfg["base_url"] or "https://api.openai.com/v1"
    elif provider_name == "groq":
        provider_cfg["base_url"] = provider_cfg["base_url"] or "https://api.groq.com/openai/v1"
    elif provider_name == "openrouter":
        provider_cfg["base_url"] = provider_cfg["base_url"] or "https://openrouter.ai/api/v1"
    elif provider_name == "moonshot":
        provider_cfg["base_url"] = provider_cfg["base_url"] or "https://api.moonshot.cn/v1"
    elif provider_name == "deepseek":
        provider_cfg["base_url"] = provider_cfg["base_url"] or "https://api.deepseek.com"
    elif provider_name == "ollama":
        provider_cfg["api_key"] = provider_cfg["api_key"] or "ollama"
        provider_cfg["base_url"] = provider_cfg["base_url"] or "http://localhost:11434/v1"

    return provider_cfg


def health_check(provider: str | None = None) -> dict[str, Any]:
    runtime = _get_provider_runtime(provider)
    missing = []
    current_provider = runtime["provider"]
    if current_provider not in {"g4f", "pollinations", "qwen", "gemini", "cloudflare", "ernie"}:
        if not runtime.get("model"):
            missing.append("model")
        if not runtime.get("base_url"):
            missing.append("base_url")
        if current_provider != "ollama" and not runtime.get("api_key"):
            missing.append("api_key")

    return {
        "provider": current_provider,
        "ok": len(missing) == 0,
        "missing": missing,
        "config": {k: v for k, v in runtime.items() if k != "api_key"},
    }


def _build_openai_client(runtime: dict[str, Any]):
    provider = runtime["provider"]
    if provider == "azure":
        return AzureOpenAI(
            api_key=runtime["api_key"],
            api_version=config.app.get("azure_api_version", "2024-02-15-preview"),
            azure_endpoint=runtime["base_url"],
        )
    return OpenAI(api_key=runtime["api_key"], base_url=runtime["base_url"])


def generate_text(prompt: str, provider: str | None = None) -> str:
    return _generate_response(prompt=prompt, provider=provider)


def generate_structured(prompt: str, provider: str | None = None, fallback_to_text: bool = True) -> dict[str, Any] | list[Any] | str:
    response = _generate_response(prompt=prompt, provider=provider)
    try:
        return json.loads(response)
    except Exception:
        if fallback_to_text:
            match = re.search(r"(\{.*\}|\[.*\])", response, re.S)
            if match:
                return json.loads(match.group(1))
            return response
        raise


def _generate_response(prompt: str, provider: str | None = None) -> str:
    try:
        content = ""
        llm_provider = (provider or config.app.get("llm_provider", "openai")).lower()
        logger.info(f"llm provider: {llm_provider}")

        if llm_provider == "g4f":
            model_name = config.app.get("g4f_model_name", "") or "gpt-3.5-turbo-16k-0613"
            content = g4f.ChatCompletion.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return content.replace("\n", "")

        if llm_provider == "pollinations":
            base_url = config.app.get("pollinations_base_url", "") or "https://text.pollinations.ai/openai"
            model_name = config.app.get("pollinations_model_name", "openai-fast")
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "seed": 101,
            }
            if config.app.get("pollinations_private"):
                payload["private"] = True
            if config.app.get("pollinations_referrer"):
                payload["referrer"] = config.app.get("pollinations_referrer")
            response = requests.post(base_url, headers={"Content-Type": "application/json"}, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].replace("\n", "")

        if llm_provider == "qwen":
            import dashscope
            from dashscope.api_entities.dashscope_response import GenerationResponse

            api_key = config.app.get("qwen_api_key")
            model_name = config.app.get("qwen_model_name")
            if not api_key or not model_name:
                raise ValueError("qwen: api_key または model_name が未設定です。")
            dashscope.api_key = api_key
            response = dashscope.Generation.call(model=model_name, messages=[{"role": "user", "content": prompt}])
            if not isinstance(response, GenerationResponse) or response.status_code != 200:
                raise Exception(f'[qwen] returned an invalid response: "{response}"')
            return response["output"]["text"].replace("\n", "")

        if llm_provider == "gemini":
            import google.generativeai as genai

            api_key = config.app.get("gemini_api_key")
            model_name = config.app.get("gemini_model_name")
            if not api_key or not model_name:
                raise ValueError("gemini: api_key または model_name が未設定です。")
            genai.configure(api_key=api_key, transport="rest")
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={"temperature": 0.5, "top_p": 1, "top_k": 1, "max_output_tokens": 2048},
            )
            response = model.generate_content(prompt)
            return response.candidates[0].content.parts[0].text.replace("\n", "")

        if llm_provider == "cloudflare":
            api_key = config.app.get("cloudflare_api_key")
            account_id = config.app.get("cloudflare_account_id")
            model_name = config.app.get("cloudflare_model_name")
            if not api_key or not account_id or not model_name:
                raise ValueError("cloudflare: account_id / api_key / model_name のいずれかが未設定です。")
            response = requests.post(
                f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"messages": [{"role": "system", "content": "You are a friendly assistant"}, {"role": "user", "content": prompt}]},
            )
            response.raise_for_status()
            return response.json()["result"]["response"]

        if llm_provider == "ernie":
            api_key = config.app.get("ernie_api_key")
            secret_key = config.app.get("ernie_secret_key")
            base_url = config.app.get("ernie_base_url")
            if not api_key or not secret_key or not base_url:
                raise ValueError("ernie: api_key / secret_key / base_url のいずれかが未設定です。")
            token_response = requests.post(
                "https://aip.baidubce.com/oauth/2.0/token",
                params={"grant_type": "client_credentials", "client_id": api_key, "client_secret": secret_key},
            )
            token_response.raise_for_status()
            access_token = token_response.json().get("access_token")
            response = requests.post(
                f"{base_url}?access_token={access_token}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(
                    {
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.5,
                        "top_p": 0.8,
                        "penalty_score": 1,
                        "disable_search": False,
                        "enable_citation": False,
                        "response_format": "text",
                    }
                ),
            )
            response.raise_for_status()
            return response.json().get("result", "")

        runtime = _get_provider_runtime(llm_provider)
        if llm_provider in OPENAI_COMPATIBLE_PROVIDERS:
            if llm_provider != "ollama" and not runtime.get("api_key"):
                raise ValueError(f"{llm_provider}: api_key が未設定です。")
            if not runtime.get("model"):
                raise ValueError(f"{llm_provider}: model が未設定です。")
            if not runtime.get("base_url"):
                raise ValueError(f"{llm_provider}: base_url が未設定です。")

        client = _build_openai_client(runtime)
        headers = None
        if llm_provider == "openrouter":
            headers = {
                "HTTP-Referer": "https://github.com/Asad-Ismail/MoneyPrinterTurbo-Extended",
                "X-Title": "MoneyPrinterTurbo-Extended",
            }

        response = client.chat.completions.create(
            model=runtime["model"],
            messages=[{"role": "user", "content": prompt}],
            extra_headers=headers,
        )
        if response and isinstance(response, ChatCompletion):
            content = response.choices[0].message.content or ""
        else:
            raise Exception(f'[{llm_provider}] returned an invalid response: "{response}"')

        return content.replace("\n", "")
    except Exception as e:
        return f"Error: {normalize_error(e)}"


def generate_script(video_subject: str, language: str = "", paragraph_number: int = 1) -> str:
    prompt = f"""
# Role: Video Script Generator

## Goals:
Generate a script for a video, depending on the subject of the video.

## Constrains:
1. the script is to be returned as a string with the specified number of paragraphs.
2. do not under any circumstance reference this prompt in your response.
3. get straight to the point, don't start with unnecessary things like, "welcome to this video".
4. you must not include any type of markdown or formatting in the script, never use a title.
5. only return the raw content of the script.
6. do not include "voiceover", "narrator" or similar indicators of what should be spoken at the beginning of each paragraph or line.
7. you must not mention the prompt, or anything about the script itself. also, never talk about the amount of paragraphs or lines. just write the script.
8. respond in the same language as the video subject.

# Initialization:
- video subject: {video_subject}
- number of paragraphs: {paragraph_number}
""".strip()
    if language:
        prompt += f"\n- language: {language}"

    final_script = ""
    logger.info(f"subject: {video_subject}")

    def format_response(response):
        response = response.replace("*", "").replace("#", "")
        response = re.sub(r"\[.*\]", "", response)
        response = re.sub(r"\(.*\)", "", response)
        return "\n\n".join(response.split("\n\n"))

    for i in range(_max_retries):
        try:
            response = _generate_response(prompt=prompt)
            if response:
                final_script = format_response(response)
            else:
                logging.error("gpt returned an empty response")
            if final_script and "当日额度已消耗完" in final_script:
                raise ValueError(final_script)
            if final_script:
                break
        except Exception as e:
            logger.error(f"failed to generate script: {e}")
        if i < _max_retries:
            logger.warning(f"failed to generate video script, trying again... {i + 1}")
    if "Error: " in final_script:
        logger.error(f"failed to generate video script: {final_script}")
    else:
        logger.success(f"completed: \n{final_script}")
    return final_script.strip()


def generate_terms(video_subject: str, video_script: str, amount: int = 5) -> List[str]:
    prompt = f"""
# Role: Video Search Terms Generator

## Goals:
Generate {amount} search terms for stock videos, depending on the subject of a video.

## Constrains:
1. the search terms are to be returned as a json-array of strings.
2. each search term should consist of 1-3 words, always add the main subject of the video.
3. you must only return the json-array of strings. you must not return anything else. you must not return the script.
4. the search terms must be related to the subject of the video.
5. reply with english search terms only.

## Output Example:
["search term 1", "search term 2", "search term 3","search term 4","search term 5"]

## Context:
### Video Subject
{video_subject}

### Video Script
{video_script}

Please note that you must use English for generating video search terms; Chinese is not accepted.
""".strip()

    logger.info(f"subject: {video_subject}")
    search_terms = []
    response = ""
    for i in range(_max_retries):
        try:
            response = _generate_response(prompt)
            if "Error: " in response:
                logger.error(f"failed to generate video script: {response}")
                return response
            search_terms = json.loads(response)
            if not isinstance(search_terms, list) or not all(isinstance(term, str) for term in search_terms):
                logger.error("response is not a list of strings.")
                continue
        except Exception as e:
            logger.warning(f"failed to generate video terms: {str(e)}")
            if response:
                match = re.search(r"\[.*]", response)
                if match:
                    try:
                        search_terms = json.loads(match.group())
                    except Exception as nested_error:
                        logger.warning(f"failed to parse fallback json: {str(nested_error)}")

        if search_terms and len(search_terms) > 0:
            break
        if i < _max_retries:
            logger.warning(f"failed to generate video terms, trying again... {i + 1}")

    logger.success(f"completed: \n{search_terms}")
    return search_terms


if __name__ == "__main__":
    video_subject = "生命的意义是什么"
    script = generate_script(video_subject=video_subject, language="zh-CN", paragraph_number=1)
    print("######################")
    print(script)
    search_terms = generate_terms(video_subject=video_subject, video_script=script, amount=5)
    print("######################")
    print(search_terms)
