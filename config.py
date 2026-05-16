import os
import yaml
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class ModelSettings(BaseModel):
    name: str = "qwen2.5-7b-instruct"
    temperature: float = 0.7
    max_tokens: int = -1
    extra_params: Dict[str, Any] = Field(default_factory=dict)

class PromptTemplate(BaseModel):
    system_message: str
    user_template: str = "Context:\n{context}\n\nTask:\n{prompt}"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")
    
    # Core Environment Variables (backwards compatible)
    openai_api_key: str = Field(default="none", validation_alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="http://localhost:11434/v1", validation_alias="OPENAI_BASE_URL")
    local_model_name: Optional[str] = Field(default=None, validation_alias="LOCAL_MODEL_NAME")
    
    # Paths (Default to absolute paths relative to this file)
    config_path: str = Field(default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    prompts_path: str = Field(default=os.path.join(os.path.dirname(__file__), "prompts.yaml"))
    usage_log_path: str = Field(default=os.path.join(os.path.dirname(__file__), "mcp_usage.jsonl"))
    
    # Internal state (loaded from YAML)
    model: ModelSettings = Field(default_factory=ModelSettings)
    prompts: Dict[str, PromptTemplate] = Field(default_factory=dict)

    def load_yaml_configs(self):
        """Load settings from YAML files if they exist."""
        # Ensure we look for .env in the project root as well
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_file):
            # Manually trigger load_dotenv to ensure env vars are populated
            # since Pydantic might have already initialized with CWD defaults
            from dotenv import load_dotenv
            load_dotenv(env_file, override=True)
            # Re-sync validated settings if needed, but Pydantic Settings 
            # will handle env vars if we re-init or use aliases correctly.
        
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f)
                if config_data and "model" in config_data:
                    self.model = ModelSettings(**config_data["model"])
        
        # Override model name if environment variable is set
        if self.local_model_name:
            self.model.name = self.local_model_name

        if os.path.exists(self.prompts_path):
            with open(self.prompts_path, "r") as f:
                prompts_data = yaml.safe_load(f)
                if prompts_data:
                    self.prompts = {
                        k: PromptTemplate(**v) for k, v in prompts_data.items()
                    }
        
        # Ensure default prompts exist
        if "general" not in self.prompts:
            self.prompts["general"] = PromptTemplate(
                system_message="You are a helpful assistant. Provide concise, accurate responses.",
                user_template="{prompt}"
            )

# Global settings instance
settings = Settings()
settings.load_yaml_configs()
