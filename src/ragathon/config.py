from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # Service Auth
    AZURE_IDENTITY_TENANT_ID: str = Field(env="AZURE_IDENTITY_TENANT_ID")
    AZURE_IDENTITY_CLIENT_ID: str = Field(env="AZURE_IDENTITY_CLIENT_ID")
    AZURE_IDENTITY_CLIENT_SECRET: str = Field(env="AZURE_IDENTITY_CLIENT_SECRET")

    # Azure Search
    AZURE_SEARCH_API_ENDPOINT: str = Field(env="AZURE_SEARCH_API_ENDPOINT")
    AZURE_SEARCH_INDEX_PREFIX: str = Field(env="AZURE_SEARCH_INDEX_PREFIX")

    # Azure OpenAI
    AZURE_OPEN_AI_API_ENDPOINT: str = Field(env="AZURE_OPEN_AI_API_ENDPOINT")
    AZURE_OPEN_AI_API_VERSION: str = Field(env="AZURE_OPEN_AI_API_VERSION")
    AZURE_OPEN_AI_EMBEDDING_DEPLOYMENT_NAME: str = Field(
        env="AZURE_OPEN_AI_EMBEDDING_DEPLOYMENT_NAME"
    )
    AZURE_OPEN_AI_EMBEDDING_MODEL_NAME: str = Field(
        env="AZURE_OPEN_AI_EMBEDDING_MODEL_NAME"
    )
    AZURE_OPEN_AI_LLM_RAG_DEPLOYMENT_NAME: str = Field(
        env="AZURE_OPEN_AI_LLM_RAG_DEPLOYMENT_NAME"
    )
    AZURE_OPEN_AI_LLM_RAG_MODEL_NAME: str = Field(
        env="AZURE_OPEN_AI_LLM_RAG_MODEL_NAME"
    )

    AZURE_OPEN_AI_LLM_EVAL_DEPLOYMENT_NAME: str = Field(
        env="AZURE_OPEN_AI_LLM_EVAL_DEPLOYMENT_NAME"
    )
    AZURE_OPEN_AI_LLM_EVAL_MODEL_NAME: str = Field(
        env="AZURE_OPEN_AI_LLM_EVAL_MODEL_NAME"
    )

    # Misc
    NLTK_DATA: str = Field(env="NLTK_DATA")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class InvalidSettingsError(RuntimeError):
    pass


def init_settings() -> Settings:
    """Initialize the application settings."""

    try:
        settings = Settings()
        return settings
    except ValidationError as e:
        msg = (
            "Error loading configuration. Please check that you have a `.env` file "
            "in the root directory of this project, and that it contains the "
            "corrent variables. Remember to follow the `.env.template` file.\n\n"
            f"Error details: {e}"
        )
        raise InvalidSettingsError(msg)
