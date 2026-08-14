"""CLI 入口 — scenelog process / scenelog search"""

import sys
from pathlib import Path

import click

from scenelog import __version__
from scenelog.config import PEOPLE_ENABLED, SCENELOG_DIR, VISION_ENABLED
from scenelog.pipeline import Pipeline


@click.group()
@click.version_option(__version__, prog_name="scenelog")
def main():
    """scenelog — 纪录片素材自动场记工具

    本地离线处理素材，生成场记表、逐字稿、全文索引，支持按对白检索。
    """
    from scenelog.logging_config import setup_logging
    setup_logging()


@main.command()
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default=None, type=click.Path(file_okay=False),
              help="输出目录（默认在素材目录下创建 _scenelog）")
@click.option("--retry-failed", is_flag=True, help="重试所有失败任务")
@click.option("--force", is_flag=True, help="忽略缓存，全部重跑")
@click.option("--rerun", default=None, type=str,
              help="仅重新生成指定步骤 (metadata/audio_extract/vad/transcription/speaker/vision/people/summary/index/excel)")
@click.option("--file", "selected_file", default=None, type=str,
              help="仅处理一个素材（相对路径或唯一文件名，建议与 --rerun 搭配）")
@click.option("--transcribe-all", is_flag=True, help="强制转录全部素材（绕过 VAD）")
@click.option("--vision/--no-vision", default=VISION_ENABLED,
              help=f"启用/禁用画面理解（默认{'开启' if VISION_ENABLED else '关闭'}）")
@click.option("--people/--no-people", default=PEOPLE_ENABLED,
              help=f"启用/禁用人物识别（默认{'开启' if PEOPLE_ENABLED else '关闭'}）")
@click.option("--dry-run", is_flag=True, help="仅扫描和预检，不实际处理")
def process(
    source_dir,
    output,
    retry_failed,
    force,
    rerun,
    selected_file,
    transcribe_all,
    vision,
    people,
    dry_run,
):
    """处理素材目录，生成场记表 + 逐字稿 + 全文索引。

    SOURCE_DIR: 素材目录路径
    """
    pipeline = Pipeline(
        source_dir=Path(source_dir).resolve(),
        output_dir=Path(output).resolve() if output else None,
        force=force,
        retry_failed=retry_failed,
        rerun_step=rerun,
        selected_file=selected_file,
        transcribe_all=transcribe_all,
        vision=vision,
        people=people,
        dry_run=dry_run,
    )
    try:
        pipeline.run()
    except KeyboardInterrupt:
        click.echo("\n用户中断，已保存进度。下次运行将自动续跑。")
        sys.exit(130)
    except Exception as e:
        click.echo(f"\n错误: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("query", type=str)
@click.option("--limit", "-n", default=20, type=int, help="最多返回条数")
@click.option("--output", "-o", default=None, type=click.Path(file_okay=False),
              help="输出目录（默认素材目录下的 _scenelog）")
def search(source_dir, query, limit, output):
    """检索素材：输入一句话，返回匹配的素材文件 + 时间点。

    SOURCE_DIR: 素材目录路径
    QUERY: 检索关键词或对白片段
    """
    from scenelog.search import Searcher

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output).resolve() if output else None

    searcher = Searcher(source_dir, output_dir)
    results = searcher.search(query, limit=limit)

    if not results:
        click.echo("未找到匹配结果。")
        return

    click.echo(f"找到 {len(results)} 条结果：\n")
    for r in results:
        source_labels = {
            "vision": "画面",
            "people": "人物",
            "identity": "身份事件",
            "audio": "语音",
        }
        source_label = source_labels.get(r.get("source"), "语音")
        click.echo(
            f"{r['rel_path']}  {r['start_time']}–{r['end_time']}  [{source_label}]"
        )
        click.echo(f'"{r["text"]}"')
        click.echo()


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="监听地址；本地使用请保持默认值")
@click.option("--port", default=8765, show_default=True, type=int,
              help="本地服务端口")
@click.option("--open/--no-open", "open_browser", default=True,
              help="启动后自动打开浏览器")
def web(host, port, open_browser):
    """启动本地网页工作台。"""
    from scenelog.web import run_server

    try:
        run_server(host=host, port=port, open_browser=open_browser)
    except OSError as e:
        raise click.ClickException(f"本地服务启动失败: {e}") from e


@main.group()
def people():
    """管理本地人物模型和人物档案。"""


@people.command("setup")
def people_setup():
    """下载 OpenCV YuNet + SFace 人物模型。"""
    from scenelog.people import install_models

    try:
        paths = install_models()
    except Exception as e:
        raise click.ClickException(f"人物模型安装失败: {e}") from e
    for path in paths:
        click.echo(f"已安装: {path}")


@people.command("voice-setup")
def people_voice_setup():
    """下载 ECAPA-TDNN 声纹模型。"""
    from scenelog.speaker import install_speaker_model

    try:
        path = install_speaker_model()
    except Exception as e:
        raise click.ClickException(f"声纹模型安装失败: {e}") from e
    click.echo(f"已安装声纹模型: {path}")


@people.command("list")
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default=None, type=click.Path(file_okay=False))
def people_list(source_dir, output):
    """列出预登记人物、参考照片数和素材命中数。"""
    store = _people_store(source_dir, output)
    profiles = store.list_people()
    if not profiles:
        click.echo("尚未登记关键人物。请先执行 scenelog people add。")
        return
    for profile in profiles:
        click.echo(
            f"{profile['id']}  {profile['name']}  "
            f"{profile['reference_count']} 张参考照片 / "
            f"{profile.get('voice_count', 0)} 段声纹 "
            f"({profile.get('voice_duration', 0):.1f} 秒) / "
            f"{profile['sample_count']} 次命中 / "
            f"{profile['material_count']} 条素材"
        )
        if profile.get("thumbnail"):
            click.echo(f"  头像: {store.output_dir / profile['thumbnail']}")


@people.command("add")
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("name")
@click.argument(
    "photos",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--output", "-o", default=None, type=click.Path(file_okay=False))
def people_add(source_dir, name, photos, output):
    """处理素材前，用一张或多张照片登记关键人物。"""
    from scenelog.people import FaceEngine

    store = _people_store(source_dir, output)
    try:
        person_id = store.add_person(name, list(photos), FaceEngine())
    except (ValueError, RuntimeError) as e:
        raise click.ClickException(str(e)) from e
    _mark_people_stale(store.output_dir)
    profile = next(
        profile
        for profile in store.list_people()
        if profile["id"] == person_id
    )
    click.echo(
        f"已登记: {person_id}  {profile['name']}  "
        f"共 {profile['reference_count']} 张参考照片"
    )
    click.echo("现在运行 scenelog process 即可；已有处理结果会自动刷新人物识别。")


@people.command("voice-add")
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("person_id")
@click.argument(
    "samples",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--output", "-o", default=None, type=click.Path(file_okay=False))
def people_voice_add(source_dir, person_id, samples, output):
    """为已登记人物追加单人声音样本。"""
    from scenelog.speaker import SpeakerEngine

    store = _people_store(source_dir, output)
    try:
        store.add_voice_samples(person_id, list(samples), SpeakerEngine())
    except (ValueError, RuntimeError) as e:
        raise click.ClickException(str(e)) from e
    _mark_speaker_stale(store.output_dir)
    profile = next(
        profile
        for profile in store.list_people()
        if profile["id"] == person_id
    )
    click.echo(
        f"已登记声纹: {profile['name']}  "
        f"{profile['voice_count']} 段 / {profile['voice_duration']:.1f} 秒"
    )


@people.command("name")
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("person_id")
@click.argument("name")
@click.option("--output", "-o", default=None, type=click.Path(file_okay=False))
def people_name(source_dir, person_id, name, output):
    """修改预登记关键人物的姓名。"""
    store = _people_store(source_dir, output)
    try:
        store.rename(person_id, name)
        _refresh_people_outputs(source_dir, output, store)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"已命名: {person_id} → {name}")


@people.command("merge")
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("source_person_id")
@click.argument("target_person_id")
@click.option("--output", "-o", default=None, type=click.Path(file_okay=False))
def people_merge(source_dir, source_person_id, target_person_id, output):
    """把 SOURCE_PERSON_ID 合并到 TARGET_PERSON_ID。"""
    store = _people_store(source_dir, output)
    try:
        store.merge(source_person_id, target_person_id)
        _refresh_people_outputs(source_dir, output, store)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"已合并: {source_person_id} → {target_person_id}")


@people.command("delete")
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("person_id")
@click.option("--output", "-o", default=None, type=click.Path(file_okay=False))
@click.confirmation_option(prompt="确定删除该人物档案？")
def people_delete(source_dir, person_id, output):
    """删除误识别的人物档案。"""
    store = _people_store(source_dir, output)
    try:
        store.delete(person_id)
        _refresh_people_outputs(source_dir, output, store)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"已删除: {person_id}")


def _people_store(source_dir, output):
    from scenelog.people import PeopleStore

    source = Path(source_dir).resolve()
    output_dir = Path(output).resolve() if output else source / SCENELOG_DIR
    return PeopleStore(output_dir)


def _mark_people_stale(output_dir: Path):
    """登记照片变化后，让下一次普通处理自动刷新人物结果。"""
    from scenelog.state import StateManager

    state = StateManager(output_dir)
    for material_id in list(state._states):
        state.invalidate_steps(
            material_id,
            ["speaker", "people", "index", "excel"],
        )


def _mark_speaker_stale(output_dir: Path):
    """声音参考变化后，只刷新声纹与依赖输出。"""
    from scenelog.state import StateManager

    state = StateManager(output_dir)
    for material_id in list(state._states):
        state.invalidate_steps(
            material_id,
            ["speaker", "index", "excel"],
        )


def _refresh_people_outputs(source_dir, output, store):
    """同步人物关系并复用管线重建人物索引和 Excel。"""
    from scenelog.state import StateManager

    source = Path(source_dir).resolve()
    output_dir = Path(output).resolve() if output else source / SCENELOG_DIR
    state = StateManager(output_dir)
    material_people = store.material_people()
    for material_id in list(state._states):
        state.set_people(material_id, material_people.get(material_id, []))

    Pipeline(
        source_dir=source,
        output_dir=output_dir,
        rerun_step="index",
        people=False,
    ).run()


if __name__ == "__main__":
    main()
