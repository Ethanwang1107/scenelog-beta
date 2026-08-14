# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


ROOT = Path(SPECPATH).parents[1]
datas = [(str(ROOT / "scenelog/web_assets"), "scenelog/web_assets")]
binaries = []
hiddenimports = collect_submodules("scenelog")

datas += collect_data_files("speechbrain", include_py_files=True)
hiddenimports += [
    "speechbrain.dataio.encoder",
    "speechbrain.inference.interfaces",
    "speechbrain.inference.speaker",
    "speechbrain.lobes.features",
    "speechbrain.lobes.models.ECAPA_TDNN",
    "speechbrain.processing.features",
    "speechbrain.utils.parameter_transfer",
]


analysis = Analysis(
    [str(ROOT / "scenelog/desktop.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="Scenelog",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=str(ROOT / "packaging/macos/entitlements.plist"),
)

collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Scenelog",
)

app = BUNDLE(
    collection,
    name="Scenelog.app",
    icon=None,
    bundle_identifier="com.scenelog.desktop",
    version="0.10.0",
    info_plist={
        "CFBundleDisplayName": "Scenelog",
        "CFBundleShortVersionString": "0.10.0",
        "CFBundleVersion": "1",
        "LSMinimumSystemVersion": "13.0",
        "NSHighResolutionCapable": True,
        "NSMicrophoneUsageDescription": "Scenelog 仅在本机读取你主动选择的声音样本，用于建立人物声纹。",
        "NSCameraUsageDescription": "Scenelog 仅在本机读取你主动选择的人物照片，用于人物识别。",
    },
)
