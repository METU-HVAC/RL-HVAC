import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def add_common_train_args(
    parser,
    *,
    season_choices: List[str],
    default_env_id: str = "A403mediumfanger",
    default_output_dir: str = "./standalone_results",
) -> None:
    parser.add_argument("--episodes", type=int, default=2, help="Number of training episodes")
    parser.add_argument("--season", type=str, default="hot", choices=season_choices)
    parser.add_argument("--env-id", type=str, default=default_env_id)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--run-tag", type=str, default=None, help="Optional short tag added to run folder name")
    parser.add_argument("--seed", type=int, default=42)


def add_common_eval_args(
    parser,
    *,
    season_choices: List[str],
    default_env_id: str = "A403mediumfanger",
    default_output_dir: str = "./eval_results",
) -> None:
    parser.add_argument("--season", type=str, default="hot", choices=season_choices)
    parser.add_argument("--env-id", type=str, default=default_env_id)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--run-tag", type=str, default=None, help="Optional short tag added to eval folder name")
    parser.add_argument("--seed", type=int, default=42)


def add_semantic_args(parser) -> None:
    """Attach shared semantic-related CLI args used by standalone scripts."""
    parser.add_argument(
        "--use-semantic",
        action="store_true",
        help="Use semantic GNN for observation augmentation",
    )
    parser.add_argument(
        "--semantic-mode",
        type=str,
        default="concat",
        choices=["concat", "latent"],
        help="'concat' = base + GNN latent, 'latent' = GNN latent only",
    )
    parser.add_argument(
        "--semantic-model-path",
        type=str,
        default=None,
        help="Optional path to a semantic graph checkpoint (.pt).",
    )
    parser.add_argument(
        "--semantic-stats-path",
        type=str,
        default=None,
        help="Optional path to semantic target stats json.",
    )


def semantic_args_metadata(args: Any, fallback_model_path: Optional[str] = None) -> Dict[str, Any]:
    """Return a serializable metadata dictionary for semantic configuration."""
    selected_model = args.semantic_model_path or fallback_model_path
    return {
        "use_semantic": bool(args.use_semantic),
        "semantic_mode": args.semantic_mode if args.use_semantic else None,
        "semantic_model_path": selected_model,
        "semantic_stats_path": args.semantic_stats_path,
    }


def build_semantic_provider(
    args: Any,
    *,
    device: str,
    semantic_root: str = "./semantic",
) -> Tuple[Optional[Any], Optional[str], Optional[str]]:
    """
    Build and return semantic provider, semantic mode, and effective model path.
    """
    if not args.use_semantic:
        return None, None, None

    model_path = args.semantic_model_path
    if model_path and not os.path.exists(model_path):
        raise FileNotFoundError(f"Semantic checkpoint not found at: {model_path}")

    stats_path = args.semantic_stats_path
    if stats_path and not os.path.exists(stats_path):
        raise FileNotFoundError(f"Semantic stats file not found at: {stats_path}")

    from semantic.source.semantic_state_provider import SemanticStateProvider

    semantic_provider = SemanticStateProvider(
        semantic_root=semantic_root,
        device=device,
        model_path=model_path,
        stats_path=stats_path,
    )
    return semantic_provider, args.semantic_mode, semantic_provider.loaded_model_path


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-_.")
    return cleaned or "na"


def semantic_run_label(
    *,
    use_semantic: bool,
    semantic_mode: Optional[str] = None,
    semantic_model_path: Optional[str] = None,
    semantic_latent_dim: Optional[int] = None,
) -> str:
    if not use_semantic:
        return "raw"

    parts = [f"semantic-{_slug(semantic_mode or 'concat')}"]
    if semantic_latent_dim is not None:
        parts.append(f"z{int(semantic_latent_dim)}")

    if semantic_model_path:
        p = Path(semantic_model_path)
        model_id = p.parent.name
        if p.parent.parent and p.parent.parent.name:
            model_id = f"{p.parent.parent.name}-{model_id}"
        parts.append(_slug(model_id))

    return "_".join(parts)


def build_train_run_name(
    *,
    algo: str,
    env_id: str,
    season: str,
    episodes: int,
    use_semantic: bool,
    semantic_mode: Optional[str] = None,
    semantic_model_path: Optional[str] = None,
    semantic_latent_dim: Optional[int] = None,
    run_tag: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    parts = [
        _slug(algo),
        _slug(env_id),
        _slug(season),
        f"ep{int(episodes)}",
        semantic_run_label(
            use_semantic=use_semantic,
            semantic_mode=semantic_mode,
            semantic_model_path=semantic_model_path,
            semantic_latent_dim=semantic_latent_dim,
        ),
    ]
    if run_tag:
        parts.append(_slug(run_tag))
    if timestamp:
        parts.append(_slug(timestamp))
    return "_".join(parts)


def build_eval_run_name(
    *,
    algo: str,
    env_id: str,
    season: str,
    model_ids: List[str],
    use_semantic: bool,
    semantic_mode: Optional[str] = None,
    semantic_model_path: Optional[str] = None,
    semantic_latent_dim: Optional[int] = None,
    run_tag: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    parts = [
        "eval",
        _slug(algo),
        _slug(env_id),
        _slug(season),
        semantic_run_label(
            use_semantic=use_semantic,
            semantic_mode=semantic_mode,
            semantic_model_path=semantic_model_path,
            semantic_latent_dim=semantic_latent_dim,
        ),
    ]
    for model_id in model_ids:
        parts.append(_slug(model_id))
    if run_tag:
        parts.append(_slug(run_tag))
    if timestamp:
        parts.append(_slug(timestamp))
    return "_".join(parts)
