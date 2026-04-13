from __future__ import annotations

from dataclasses import dataclass


COLLECTIBLE_KIND_MINERAL = "mineral"
COLLECTIBLE_KIND_PLANT = "plant"


@dataclass(frozen=True)
class CollectibleMaterial:
    category_id: int
    name: str
    kind: str


# biliwiki “大地图 -> 采集” 当前可用的 34 种素材：
# 714 / 715 页面不存在，720 当前为空数组，因此不纳入可选清单。
ACTIVE_COLLECTIBLE_MATERIALS: tuple[CollectibleMaterial, ...] = (
    CollectibleMaterial(701, "黑晶琉璃", COLLECTIBLE_KIND_MINERAL),
    CollectibleMaterial(702, "黄石榴石", COLLECTIBLE_KIND_MINERAL),
    CollectibleMaterial(703, "蓝晶碧玺", COLLECTIBLE_KIND_MINERAL),
    CollectibleMaterial(704, "紫莲刚玉", COLLECTIBLE_KIND_MINERAL),
    CollectibleMaterial(705, "向阳花", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(706, "喵喵草", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(707, "蓝掌", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(708, "睡铃", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(709, "天使草", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(710, "石耳", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(711, "伞伞菌", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(712, "蜜黄菌", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(713, "喷气菇", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(716, "星霜花", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(717, "荧光兰", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(718, "大嘴花", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(719, "流星兰", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(721, "海桑花", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(722, "海星石", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(723, "彩玉花", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(724, "象牙花", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(725, "风卷草", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(726, "海珊瑚", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(727, "海神花", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(728, "紫雀花", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(729, "恶魔雪茄", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(730, "骨片", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(731, "花星角", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(732, "火焰花", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(733, "雪菇", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(734, "幽幽草", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(735, "幽幽鬼火", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(736, "藻羽花", COLLECTIBLE_KIND_PLANT),
    CollectibleMaterial(737, "洋红珊瑚", COLLECTIBLE_KIND_PLANT),
)

COLLECTIBLE_MATERIALS_BY_ID = {material.category_id: material for material in ACTIVE_COLLECTIBLE_MATERIALS}
ACTIVE_COLLECTIBLE_IDS = tuple(material.category_id for material in ACTIVE_COLLECTIBLE_MATERIALS)
MINERAL_COLLECTIBLE_IDS = tuple(
    material.category_id for material in ACTIVE_COLLECTIBLE_MATERIALS if material.kind == COLLECTIBLE_KIND_MINERAL
)
PLANT_COLLECTIBLE_IDS = tuple(
    material.category_id for material in ACTIVE_COLLECTIBLE_MATERIALS if material.kind == COLLECTIBLE_KIND_PLANT
)


def active_collectible_materials() -> tuple[CollectibleMaterial, ...]:
    return ACTIVE_COLLECTIBLE_MATERIALS


def collectible_material_by_id(category_id: int) -> CollectibleMaterial | None:
    return COLLECTIBLE_MATERIALS_BY_ID.get(int(category_id))


def collectible_ids_for_kind(kind: str | None = None) -> tuple[int, ...]:
    if kind == COLLECTIBLE_KIND_MINERAL:
        return MINERAL_COLLECTIBLE_IDS
    if kind == COLLECTIBLE_KIND_PLANT:
        return PLANT_COLLECTIBLE_IDS
    return ACTIVE_COLLECTIBLE_IDS
