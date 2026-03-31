import pytest
from unittest.mock import patch, MagicMock
from rocinsight.ai_analysis.llm_analyzer import LLMAnalyzer


class TestLocalLLM:
    def test_local_provider_accepted(self):
        a = LLMAnalyzer(provider="local", model="codellama:13b", api_key="ignored")
        assert a.provider == "local"

    def test_summarize_source_file_calls_local(self):
        a = LLMAnalyzer(provider="local", model="codellama:13b", api_key="ignored")
        with patch.object(a, "_call_local", return_value="summary text") as mock:
            result = a.summarize_source_file("kernel.hip", "// hip kernel code")
        mock.assert_called_once()
        assert result == "summary text"

    def test_annotate_profiling_plan_calls_online(self):
        a = LLMAnalyzer(provider="anthropic", api_key="sk-ant-test")
        with patch.object(a, "_call_anthropic", return_value="advice") as mock:
            result = a.annotate_profiling_plan({"kernel_count": 3})
        mock.assert_called_once()
        assert result == "advice"

    def test_call_local_uses_openai_compat_endpoint(self):
        pytest.importorskip("openai", reason="openai package not installed")
        a = LLMAnalyzer(provider="local", model="codellama:13b", api_key="ignored")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        with patch("openai.OpenAI") as MockClient:
            instance = MockClient.return_value
            instance.chat.completions.create.return_value = mock_resp
            result = a._call_local("sys", "user")
        assert result == "ok"
        call_kwargs = MockClient.call_args[1]
        assert "localhost" in call_kwargs.get("base_url", "")
