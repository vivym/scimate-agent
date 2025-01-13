from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "INFO"

    secret_key: str = "secret"
    secret_algorithm: str = "HS256"
    access_token_expire_minutes: int = 24 * 60


settings = Settings(
    _env_file=".env",
    _env_file_encoding="utf-8",
)
