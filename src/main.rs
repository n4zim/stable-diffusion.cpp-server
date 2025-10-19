use actix_web::{middleware, web, App, HttpRequest, HttpResponse, HttpServer};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::process::Command;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
  let context = Context::default();
  let port = context.port;

  println!("Starting stable-diffusion.cpp server on port {}...", port);

  HttpServer::new(move || {
    App::new()
      .app_data(web::Data::new(context.clone()))
      .wrap(middleware::Logger::default())
      .route("/v1/images/generations", web::post().to(generate_image))
      .route("/health", web::get().to(health_check))
  })
  .bind(("0.0.0.0", port))?
  .run()
  .await
}

#[derive(Clone)]
struct Context {
  port: u16,
  token: String,
  binary_path: String,
  args: Option<Vec<String>>,
  models_dir: String,
  cache_dir: String,
}

impl Default for Context {
  fn default() -> Self {
    Context {
      port: std::env::var("SD_CPP_SERVER_PORT")
        .expect("SD_CPP_SERVER_PORT environment variable not set")
        .parse::<u16>()
        .expect("SD_CPP_SERVER_PORT must be a valid port number"),
      token: std::env::var("SD_CPP_SERVER_TOKEN")
        .expect("SD_CPP_SERVER_TOKEN environment variable not set"),
      binary_path: std::env::var("SD_CPP_SERVER_BINARY")
        .expect("SD_CPP_SERVER_BINARY environment variable not set"),
      args: std::env::var("SD_CPP_SERVER_ARGS")
        .ok()
        .map(|s| s.split_whitespace().map(|s| s.to_string()).collect()),
      models_dir: std::env::var("SD_CPP_SERVER_MODELS")
        .expect("SD_CPP_SERVER_MODELS environment variable not set"),
      cache_dir: std::env::var("SD_CPP_SERVER_CACHE")
        .unwrap_or_else(|_| "/tmp".to_string()),
    }
  }
}

async fn generate_image(
  req: HttpRequest,
  body: web::Json<ImageGenerationRequest>,
  context: web::Data<Context>,
) -> HttpResponse {
  if let Err(response) = verify_bearer_token(&req, &context.token) {
    return response;
  }

  let timestamp = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap()
    .as_secs();

  let output_path =
    format!("{}/sd_output_{}.png", context.cache_dir, timestamp);

  let mut cmd = Command::new(&context.binary_path);
  if let Some(args) = &context.args {
    for arg in args {
      cmd.arg(arg);
    }
  }
  cmd
    .arg("-m")
    .arg(format!("{}/{}", context.models_dir, body.model));
  cmd.arg("-p").arg(&body.prompt);
  cmd.arg("-o").arg(&output_path);
  cmd.arg("--steps").arg(body.steps.to_string());
  cmd.arg("--cfg-scale").arg(body.cfg_scale.to_string());

  if body.seed >= 0 {
    cmd.arg("--seed").arg(body.seed.to_string());
  }

  if let Some(neg_prompt) = &body.negative_prompt {
    cmd.arg("-n").arg(neg_prompt);
  }

  let size_parts: Vec<&str> = body.size.split('x').collect();
  if size_parts.len() == 2 {
    cmd.arg("-W").arg(size_parts[0]);
    cmd.arg("-H").arg(size_parts[1]);
  }

  match cmd.output().await {
    Ok(output) => {
      if output.status.success() {
        match tokio::fs::read(&output_path).await {
          Ok(image_data) => {
            let b64 = base64::Engine::encode(
              &base64::engine::general_purpose::STANDARD,
              &image_data,
            );
            let _ = tokio::fs::remove_file(&output_path).await;
            HttpResponse::Ok().json(ImageGenerationResponse {
              created: timestamp,
              data: vec![ImageData { b64_json: b64 }],
            })
          }
          Err(e) => HttpResponse::InternalServerError().json(ErrorResponse {
            error: ErrorDetail {
              message: format!("Failed to read output image: {}", e),
              error_type: "server_error".to_string(),
            },
          }),
        }
      } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        HttpResponse::InternalServerError().json(ErrorResponse {
          error: ErrorDetail {
            message: format!("Image generation failed: {}", stderr),
            error_type: "server_error".to_string(),
          },
        })
      }
    }
    Err(e) => HttpResponse::InternalServerError().json(ErrorResponse {
      error: ErrorDetail {
        message: format!("Failed to execute sd command: {}", e),
        error_type: "server_error".to_string(),
      },
    }),
  }
}

#[derive(Debug, Deserialize)]
struct ImageGenerationRequest {
  prompt: String,
  model: String,
  #[serde(default = "default_size")]
  size: String,
  #[serde(default)]
  negative_prompt: Option<String>,
  #[serde(default = "default_steps")]
  steps: u32,
  #[serde(default = "default_cfg_scale")]
  cfg_scale: f32,
  #[serde(default = "default_seed")]
  seed: i32,
}

fn default_size() -> String {
  "512x512".to_string()
}

fn default_steps() -> u32 {
  20
}

fn default_cfg_scale() -> f32 {
  7.0
}

fn default_seed() -> i32 {
  -1
}

#[derive(Debug, Serialize)]
struct ImageGenerationResponse {
  created: u64,
  data: Vec<ImageData>,
}

#[derive(Debug, Serialize)]
struct ImageData {
  b64_json: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
  error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
  message: String,
  #[serde(rename = "type")]
  error_type: String,
}

fn verify_bearer_token(
  req: &HttpRequest,
  expected_token: &str,
) -> Result<(), HttpResponse> {
  if let Some(auth_header) = req.headers().get("authorization") {
    if let Ok(auth_str) = auth_header.to_str() {
      if auth_str.starts_with("Bearer ") && &auth_str[7..] == expected_token {
        return Ok(());
      }
    }
  }
  Err(HttpResponse::Unauthorized().json(ErrorResponse {
    error: ErrorDetail {
      message: "Invalid or missing authorization token".to_string(),
      error_type: "invalid_request_error".to_string(),
    },
  }))
}

async fn health_check() -> HttpResponse {
  HttpResponse::Ok().json(serde_json::json!({
      "status": "ok",
      "timestamp": SystemTime::now()
          .duration_since(UNIX_EPOCH)
          .unwrap()
          .as_secs()
  }))
}
