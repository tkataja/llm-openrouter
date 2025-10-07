import click
from enum import Enum
import llm
from llm.default_plugins.openai_models import Chat, AsyncChat
from pathlib import Path
from pydantic import Field, field_validator
from typing import Optional, Union
import json
import time
import httpx


def get_openrouter_models():
    models = fetch_cached_json(
        url="https://openrouter.ai/api/v1/models",
        path=llm.user_dir() / "openrouter_models.json",
        cache_timeout=3600,
    )["data"]
    return models


def get_supports_images(model_definition):
    try:
        return "image" in model_definition["architecture"]["input_modalities"]
    except KeyError:
        return False


def has_parameter(model_definition, parameter):
    try:
        return parameter in model_definition["supported_parameters"]
    except KeyError:
        return False


class ReasoningEffortEnum(str, Enum):
    minimal = "minimal"
    low = "low"
    medium = "medium"
    high = "high"


class VerbosityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class SearchEngineEnum(str, Enum):
    native = "native"
    exa = "exa"


class SearchContextSizeEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class _mixin:
    class Options(Chat.Options):
        online: Optional[bool] = Field(
            description="Use relevant search results from Exa (deprecated, use plugins instead)",
            default=None,
        )
        plugins: Optional[Union[dict, str]] = Field(
            description='JSON object or string to configure plugins, e.g. {"id": "web", "engine": "exa", "max_results": 5}',
            default=None,
        )
        web_search_engine: Optional[SearchEngineEnum] = Field(
            description='Search engine for web plugin: "native" or "exa"',
            default=None,
        )
        web_search_max_results: Optional[int] = Field(
            description="Number of search results to return (default: 5)",
            default=None,
        )
        web_search_prompt: Optional[str] = Field(
            description="Custom prompt for integrating search results",
            default=None,
        )
        web_search_context_size: Optional[SearchContextSizeEnum] = Field(
            description='Search context size: "low", "medium", or "high"',
            default=None,
        )
        provider: Optional[Union[dict, str]] = Field(
            description=("JSON object to control provider routing"),
            default=None,
        )
        reasoning_effort: Optional[ReasoningEffortEnum] = Field(
            description='One of "high", "medium", "low" or "minimal" to control reasoning effort',
            default=None,
        )
        reasoning_max_tokens: Optional[int] = Field(
            description="Specific token limit to control reasoning effort",
            default=None,
        )
        reasoning_enabled: Optional[bool] = Field(
            description="Set to true to enable reasoning with default parameters",
            default=None,
        )
        verbosity: Optional[VerbosityEnum] = Field(
            description='One of "high", "medium", "low", "minimal" to control verbosity',
            default=None,
        )

        @field_validator("provider")
        def validate_provider(cls, provider):
            if provider is None:
                return None

            if isinstance(provider, str):
                try:
                    return json.loads(provider)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON in provider string")
            return provider

        @field_validator("plugins")
        def validate_plugins(cls, plugins):
            if plugins is None:
                return None

            if isinstance(plugins, str):
                try:
                    return json.loads(plugins)
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON in plugins string")
            return plugins

    def build_kwargs(self, prompt, stream):
        kwargs = super().build_kwargs(prompt, stream)
        # Remove custom options from kwargs (they go in extra_body)
        kwargs.pop("provider", None)
        kwargs.pop("online", None)
        kwargs.pop("plugins", None)
        kwargs.pop("web_search_engine", None)
        kwargs.pop("web_search_max_results", None)
        kwargs.pop("web_search_prompt", None)
        kwargs.pop("web_search_context_size", None)
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("reasoning_max_tokens", None)
        kwargs.pop("reasoning_enabled", None)
        kwargs.pop("verbosity", None)

        extra_body = {}

        # Handle web search plugins
        if prompt.options.plugins:
            # User provided explicit plugins config
            extra_body["plugins"] = prompt.options.plugins if isinstance(prompt.options.plugins, list) else [prompt.options.plugins]
        elif prompt.options.online or any([
            prompt.options.web_search_engine,
            prompt.options.web_search_max_results,
            prompt.options.web_search_prompt,
        ]):
            # Build web plugin config from individual options
            web_plugin = {"id": "web"}
            if prompt.options.web_search_engine:
                web_plugin["engine"] = prompt.options.web_search_engine
            if prompt.options.web_search_max_results:
                web_plugin["max_results"] = prompt.options.web_search_max_results
            if prompt.options.web_search_prompt:
                web_plugin["search_prompt"] = prompt.options.web_search_prompt
            extra_body["plugins"] = [web_plugin]

        # Handle web_search_options
        if prompt.options.web_search_context_size:
            extra_body["web_search_options"] = {
                "search_context_size": prompt.options.web_search_context_size
            }

        # Handle provider routing
        if prompt.options.provider:
            extra_body["provider"] = prompt.options.provider

        # Handle reasoning parameters
        reasoning = {}
        if prompt.options.reasoning_effort:
            reasoning["effort"] = prompt.options.reasoning_effort
        if prompt.options.reasoning_max_tokens:
            reasoning["max_tokens"] = prompt.options.reasoning_max_tokens
        if prompt.options.reasoning_enabled is not None:
            reasoning["enabled"] = prompt.options.reasoning_enabled
        if reasoning:
            extra_body["reasoning"] = reasoning

        # Handle verbosity
        if prompt.options.verbosity is not None:
            extra_body["verbosity"] = prompt.options.verbosity

        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs


class OpenRouterChat(_mixin, Chat):
    needs_key = "openrouter"
    key_env_var = "OPENROUTER_KEY"

    def __str__(self):
        return "OpenRouter: {}".format(self.model_id)


class OpenRouterAsyncChat(_mixin, AsyncChat):
    needs_key = "openrouter"
    key_env_var = "OPENROUTER_KEY"

    def __str__(self):
        return "OpenRouter: {}".format(self.model_id)


@llm.hookimpl
def register_models(register):
    # Only do this if the openrouter key is set
    key = llm.get_key("", "openrouter", "OPENROUTER_KEY")
    if not key:
        return
    for model_definition in get_openrouter_models():
        supports_images = get_supports_images(model_definition)
        kwargs = dict(
            model_id="openrouter/{}".format(model_definition["id"]),
            model_name=model_definition["id"],
            vision=supports_images,
            supports_schema=has_parameter(model_definition, "structured_outputs"),
            supports_tools=has_parameter(model_definition, "tools"),
            api_base="https://openrouter.ai/api/v1",
            headers={"HTTP-Referer": "https://llm.datasette.io/", "X-Title": "LLM"},
        )
        register(
            OpenRouterChat(**kwargs),
            OpenRouterAsyncChat(**kwargs),
        )


class DownloadError(Exception):
    pass


def fetch_cached_json(url, path, cache_timeout):
    path = Path(path)

    # Create directories if not exist
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file():
        # Get the file's modification time
        mod_time = path.stat().st_mtime
        # Check if it's more than the cache_timeout old
        if time.time() - mod_time < cache_timeout:
            # If not, load the file
            with open(path, "r") as file:
                return json.load(file)

    # Try to download the data
    try:
        response = httpx.get(url, follow_redirects=True)
        response.raise_for_status()  # This will raise an HTTPError if the request fails

        # If successful, write to the file
        with open(path, "w") as file:
            json.dump(response.json(), file)

        return response.json()
    except httpx.HTTPError:
        # If there's an existing file, load it
        if path.is_file():
            with open(path, "r") as file:
                return json.load(file)
        else:
            # If not, raise an error
            raise DownloadError(
                f"Failed to download data and no cache is available at {path}"
            )


@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def openrouter():
        "Commands relating to the llm-openrouter plugin"

    @openrouter.command()
    @click.option("--free", is_flag=True, help="List free models")
    @click.option("json_", "--json", is_flag=True, help="Output as JSON")
    def models(free, json_):
        "List of OpenRouter models"
        if free:
            all_models = [
                model
                for model in get_openrouter_models()
                if model["id"].endswith(":free")
            ]
        else:
            all_models = get_openrouter_models()
        if json_:
            click.echo(json.dumps(all_models, indent=2))
        else:
            # Custom format
            for model in all_models:
                bits = []
                bits.append(f"- id: {model['id']}")
                bits.append(f"  name: {model['name']}")
                bits.append(f"  context_length: {model['context_length']:,}")
                architecture = model.get("architecture", None)
                if architecture:
                    bits.append("  architecture:")
                    for key, value in architecture.items():
                        bits.append(
                            "    "
                            + key
                            + ": "
                            + (value if isinstance(value, str) else json.dumps(value))
                        )
                bits.append(f"  supports_schema: {has_parameter(model, 'structured_outputs')}")
                bits.append(f"  supports_tools: {has_parameter(model, 'tools')}")
                pricing = format_pricing(model["pricing"])
                if pricing:
                    bits.append("  pricing: " + pricing)
                click.echo("\n".join(bits) + "\n")

    @openrouter.command()
    @click.option("--key", help="Key to inspect")
    def key(key):
        "View information and rate limits for the current key"
        key = llm.get_key(key, "openrouter", "OPENROUTER_KEY")
        response = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {key}"},
        )
        response.raise_for_status()
        click.echo(json.dumps(response.json()["data"], indent=2))


def format_price(key, price_str):
    """Format a price value with appropriate scaling and no trailing zeros."""
    price = float(price_str)

    if price == 0:
        return None

    # Determine scale based on magnitude
    if price < 0.0001:
        scale = 1000000
        suffix = "/M"
    elif price < 0.001:
        scale = 1000
        suffix = "/K"
    elif price < 1:
        scale = 1000
        suffix = "/K"
    else:
        scale = 1
        suffix = ""

    # Scale the price
    scaled_price = price * scale

    # Format without trailing zeros
    # Convert to string and remove trailing .0
    price_str = (
        f"{scaled_price:.10f}".rstrip("0").rstrip(".")
        if "." in f"{scaled_price:.10f}"
        else f"{scaled_price:.0f}"
    )

    return f"{key} ${price_str}{suffix}"


def format_pricing(pricing_dict):
    formatted_parts = []
    for key, value in pricing_dict.items():
        formatted_price = format_price(key, value)
        if formatted_price:
            formatted_parts.append(formatted_price)
    return ", ".join(formatted_parts)
