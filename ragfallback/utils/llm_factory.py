"""Factory functions for creating LLMs (open-source and paid)."""

import os
from typing import Optional

from langchain_core.language_models import BaseLanguageModel

from ragfallback.utils.env import load_env, mistral_config_from_env


def create_open_source_llm(
    model: str = "llama3",
    base_url: Optional[str] = None,
    temperature: float = 0,
    provider: str = "ollama"
) -> BaseLanguageModel:
    """
    Create an open-source LLM using Ollama (default) or HuggingFace.
    
    Args:
        model: Model name
               For Ollama: "llama3", "llama2", "mistral", "codellama", etc.
               For HuggingFace: "mistralai/Mistral-7B-Instruct-v0.1", etc.
        base_url: Ollama base URL (default: "http://localhost:11434") - only for Ollama
        temperature: Temperature for generation (default: 0)
        provider: Provider to use - "ollama" (default) or "huggingface"
    
    Returns:
        BaseLanguageModel instance
        
    Example:
        >>> # Using Ollama (requires Ollama installed)
        >>> from ragfallback.utils.llm_factory import create_open_source_llm
        >>> llm = create_open_source_llm(model="llama3", provider="ollama")
        
        >>> # Using HuggingFace Inference API (no installation needed, free tier)
        >>> llm = create_open_source_llm(
        ...     model="mistralai/Mistral-7B-Instruct-v0.1",
        ...     provider="huggingface"
        ... )
    """
    if provider.lower() == "huggingface":
        return create_huggingface_llm(
            model_id=model,
            temperature=temperature,
            use_inference_api=True  # Use Inference API by default (easier)
        )
    else:
        # Default to Ollama
        try:
            from langchain_community.llms import Ollama
            
            base_url = base_url or "http://localhost:11434"
            return Ollama(
                model=model,
                base_url=base_url,
                temperature=temperature
            )
        except ImportError:
            raise ImportError(
                "Ollama not installed. Install with: pip install langchain-community. "
                "Or use provider='huggingface' for HuggingFace Inference API."
            )


def create_huggingface_llm(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.1",
    device: str = "cpu",
    temperature: float = 0,
    use_inference_api: bool = False,
    api_key: Optional[str] = None,
    max_length: int = 512
) -> BaseLanguageModel:
    """
    Create a HuggingFace LLM using transformers (fully open-source).
    
    Supports two modes:
    1. Local models (use_inference_api=False) - Runs models locally
    2. Inference API (use_inference_api=True) - Uses HuggingFace hosted API (free tier available)
    
    Args:
        model_id: HuggingFace model ID
                  Popular options:
                  - "mistralai/Mistral-7B-Instruct-v0.1"
                  - "meta-llama/Llama-2-7b-chat-hf"
                  - "microsoft/DialoGPT-medium"
                  - "google/flan-t5-base"
        device: Device to run on ("cpu" or "cuda") - only for local mode
        temperature: Temperature for generation
        use_inference_api: If True, use HuggingFace Inference API (free tier available)
                          If False, load model locally (requires more memory)
        api_key: HuggingFace API key (optional, uses HUGGINGFACE_API_KEY env var)
                 Required for Inference API with private models
        max_length: Maximum generation length
    
    Returns:
        BaseLanguageModel instance
        
    Example:
        >>> # Using Inference API (easier, free tier available)
        >>> from ragfallback.utils.llm_factory import create_huggingface_llm
        >>> llm = create_huggingface_llm(
        ...     model_id="mistralai/Mistral-7B-Instruct-v0.1",
        ...     use_inference_api=True
        ... )
        
        >>> # Using local model (requires transformers and torch)
        >>> llm = create_huggingface_llm(
        ...     model_id="google/flan-t5-base",
        ...     use_inference_api=False,
        ...     device="cpu"
        ... )
    """
    if use_inference_api:
        # Use HuggingFace Inference API (easier, free tier available)
        try:
            from langchain_community.llms import HuggingFaceEndpoint
            
            return HuggingFaceEndpoint(
                endpoint_url=f"https://api-inference.huggingface.co/pipeline/text-generation/{model_id}",
                huggingfacehub_api_token=api_key,
                task="text-generation",
                temperature=temperature,
                model_kwargs={
                    "max_length": max_length,
                }
            )
        except ImportError:
            raise ImportError(
                "HuggingFace Inference API not available. "
                "Install with: pip install huggingface-hub"
            )
    else:
        # Use local transformers model
        try:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype="auto" if device == "cuda" else None
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
                model_kwargs={
                    "temperature": temperature,
                    "max_length": max_length,
                }
            )
            
            return HuggingFacePipeline(pipeline=pipe)
        except ImportError:
            raise ImportError(
                "HuggingFace transformers not installed. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HuggingFace model {model_id}. "
                f"Error: {str(e)}. "
                f"Try using use_inference_api=True for easier setup."
            )


def create_mistral_llm(
    model: Optional[str] = None,
    temperature: float = 0,
    api_key: Optional[str] = None,
    load_dotenv: bool = True,
) -> BaseLanguageModel:
    """
    Create a Mistral chat model (API). Uses ``MISTRAL_API_KEY`` and ``MISTRAL_MODEL``
    from the environment by default.

    Install: ``pip install langchain-mistralai`` (and ``python-dotenv`` to load ``.env``).

    Args:
        model: Override model id (default: ``MISTRAL_MODEL`` or ``mistral-small-latest``).
        temperature: Sampling temperature.
        api_key: Override API key (default: ``MISTRAL_API_KEY``).
        load_dotenv: If True, call :func:`~ragfallback.utils.env.load_env` first.
    """
    if load_dotenv:
        load_env()
    key = api_key or os.environ.get("MISTRAL_API_KEY") or os.environ.get(
        "MISTRAL_API_TOKEN"
    )
    if not key:
        raise ValueError(
            "Mistral API key missing. Set MISTRAL_API_KEY in your environment or .env "
            "(see .env.example)."
        )
    mid = model or os.environ.get("MISTRAL_MODEL", "mistral-small-latest")
    try:
        from langchain_mistralai import ChatMistralAI
    except ImportError as e:
        raise ImportError(
            "Mistral LLM requires langchain-mistralai. "
            "Install with: pip install langchain-mistralai"
        ) from e
    return ChatMistralAI(model=mid, api_key=key, temperature=temperature)


def create_llm_from_env(
    temperature: float = 0,
    load_dotenv: bool = True,
) -> BaseLanguageModel:
    """
    Preferred LLM for apps using this repo's ``.env``: Mistral when ``MISTRAL_API_KEY`` is set.

    Raises:
        ValueError: if no Mistral key is available after loading ``.env``.
    """
    if load_dotenv:
        load_env()
    key, model = mistral_config_from_env()
    if not key:
        raise ValueError(
            "MISTRAL_API_KEY not set. Add it to .env or the environment, or use "
            "create_open_source_llm / create_huggingface_llm / create_openai_llm explicitly."
        )
    return create_mistral_llm(model=model, temperature=temperature, api_key=key, load_dotenv=False)


def create_openai_llm(
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    api_key: Optional[str] = None
) -> BaseLanguageModel:
    """
    Create an OpenAI LLM (paid, requires API key).
    
    Args:
        model: OpenAI model name
        temperature: Temperature for generation
        api_key: OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)
    
    Returns:
        BaseLanguageModel instance
        
    Example:
        >>> from ragfallback.utils.llm_factory import create_openai_llm
        >>> llm = create_openai_llm(model="gpt-4o-mini")
    """
    try:
        from langchain_openai import ChatOpenAI
        
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
    except ImportError:
        raise ImportError(
            "OpenAI not installed. Install with: pip install langchain-openai"
        )


def create_anthropic_llm(
    model: str = "claude-3-haiku-20240307",
    temperature: float = 0,
    api_key: Optional[str] = None
) -> BaseLanguageModel:
    """
    Create an Anthropic Claude LLM (paid, requires API key).
    
    Args:
        model: Anthropic model name
        temperature: Temperature for generation
        api_key: Anthropic API key (optional, uses ANTHROPIC_API_KEY env var if not provided)
    
    Returns:
        BaseLanguageModel instance
    """
    try:
        from langchain_anthropic import ChatAnthropic
        
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
    except ImportError:
        raise ImportError(
            "Anthropic not installed. Install with: pip install langchain-anthropic"
        )

