from blockrun_litellm.proxy import _chat_payload_to_response


def test_chat_to_responses_preserves_token_details() -> None:
    response = _chat_payload_to_response(
        {
            "id": "chatcmpl-1",
            "model": "openai/gpt-5.5",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 12},
            },
        },
        "openai/gpt-5.5",
    )

    assert response["usage"]["input_tokens"] == 10
    assert response["usage"]["output_tokens"] == 20
    assert response["usage"]["input_tokens_details"]["cached_tokens"] == 4
    assert response["usage"]["output_tokens_details"]["reasoning_tokens"] == 12
