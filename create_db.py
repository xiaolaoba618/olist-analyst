# -*- coding: utf-8 -*-
"""
create_db.py
------------
把 data/ 目录下的多个 Olist CSV 文件导入到一个 SQLite 数据库 olist_db 中。

输出位置：data/database/olist_db.db
之后即可用标准库 sqlite3 直接写 SQL 查询库中的表。

运行方式（使用 D:\\anaconda3\\envs\\DA_TOOL 环境）：
    D:\\anaconda3\\envs\\DA_TOOL\\python.exe create_db.py
"""

import os
import sqlite3
import time
from pathlib import Path

import pandas as pd

# ============================== 配置区 ==============================

# 项目根目录（本脚本所在目录）
BASE_DIR = Path(__file__).resolve().parent

# CSV 所在目录
DATA_DIR = BASE_DIR  / "data" / "raw"

# 数据库输出目录与文件名
DB_DIR = BASE_DIR / "db"
DB_NAME = "olist"                 # 逻辑库名
DB_FILE = DB_DIR / f"{DB_NAME}.db"   # 实际 SQLite 文件

# 编码兜底：先 utf-8，失败再 latin-1
ENCODINGS = ("utf-8", "latin-1")

# 大文件分块写入的行数（避免一次性占满内存）
CHUNK_SIZE = 200_000

# ====================== 表名映射（可选自定义） ======================
# 默认规则：去掉 "olist_" 前缀 + "_dataset" 后缀
#   例：olist_order_items_dataset.csv -> order_items
TABLE_NAME_MAP = {
    "olist_customers_dataset.csv": "customers",
    "olist_geolocation_dataset.csv": "geolocation",
    "olist_order_items_dataset.csv": "order_items",
    "olist_order_payments_dataset.csv": "order_payments",
    "olist_order_reviews_dataset.csv": "order_reviews",
    "olist_orders_dataset.csv": "orders",
    "olist_products_dataset.csv": "products",
    "olist_sellers_dataset.csv": "sellers",
    "product_category_name_translation.csv": "product_category_name_translation",
}

# 需要建立索引的字段（加速 JOIN / 过滤）
INDEXES = {
    "customers": ["customer_id", "customer_unique_id", "customer_zip_code_prefix"],
    "geolocation": ["geolocation_zip_code_prefix"],
    "order_items": ["order_id", "product_id", "seller_id"],
    "order_payments": ["order_id"],
    "order_reviews": ["order_id", "review_id"],
    "orders": ["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
    "products": ["product_id", "product_category_name"],
    "sellers": ["seller_id", "seller_zip_code_prefix"],
    "product_category_name_translation": ["product_category_name"],
}

# ============================== 函数区 ==============================


def derive_table_name(csv_path: Path) -> str:
    """根据文件名推导表名：去前缀 olist_ 与后缀 _dataset。"""
    if csv_path.name in TABLE_NAME_MAP:
        return TABLE_NAME_MAP[csv_path.name]
    stem = csv_path.stem
    if stem.startswith("olist_"):
        stem = stem[len("olist_"):]
    if stem.endswith("_dataset"):
        stem = stem[: -len("_dataset")]
    return stem


def read_csv_safely(csv_path: Path) -> pd.DataFrame:
    """带编码兜底地读取 CSV。"""
    last_err = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(csv_path, encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def import_csv_to_sqlite(csv_path: Path, conn: sqlite3.Connection, table: str) -> int:
    """把单个 CSV 写入 SQLite，返回写入行数。"""
    df = read_csv_safely(csv_path)

    # 列名统一小写、去空格，避免 SQL 书写踩坑
    df.columns = [c.strip().lower() for c in df.columns]

    if len(df) > CHUNK_SIZE:
        # 大表分块写入
        df.head(0).to_sql(table, conn, if_exists="replace", index=False)
        for start in range(0, len(df), CHUNK_SIZE):
            df.iloc[start: start + CHUNK_SIZE].to_sql(
                table, conn, if_exists="append", index=False
            )
    else:
        df.to_sql(table, conn, if_exists="replace", index=False)

    return len(df)


def create_indexes(conn: sqlite3.Connection, table: str) -> None:
    """为常用字段建索引（字段不存在则跳过）。"""
    cols = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
    for col in INDEXES.get(table, []):
        if col not in cols:
            continue
        idx_name = f"idx_{table}_{col}"
        conn.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table}"("{col}")')
    conn.commit()


def verify(conn: sqlite3.Connection) -> None:
    """验证：列出所有表 + 行数 + 跑一条示例 SQL。"""
    print("\n" + "=" * 60)
    print("数据库内容校验")
    print("=" * 60)

    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]

    print(f"{'表名':<38}{'行数':>12}   字段数")
    print("-" * 60)
    for t in tables:
        cnt = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        col_cnt = len(conn.execute(f'PRAGMA table_info("{t}")').fetchall())
        print(f"{t:<38}{cnt:>12,}   {col_cnt}")

    print("\n示例查询：各订单状态数量（TOP 10）")
    print("-" * 60)
    sql = """
        SELECT order_status,
               COUNT(*) AS order_cnt
        FROM orders
        GROUP BY order_status
        ORDER BY order_cnt DESC
        LIMIT 10
    """
    for row in conn.execute(sql):
        print(f"  {row[0]:<20}{row[1]:>10,}")


# ============================== 主流程 ==============================

def main() -> None:
    start = time.time()

    print("=" * 60)
    print("Olist CSV -> SQLite 导入工具")
    print("=" * 60)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"找不到数据目录：{DATA_DIR}")

    # 1. 创建输出目录
    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"数据目录：{DATA_DIR}")
    print(f"数据库文件：{DB_FILE}\n")

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"{DATA_DIR} 下没有找到任何 CSV 文件")

    # 2. 若旧库存在则先删除，保证重建干净
    if DB_FILE.exists():
        DB_FILE.unlink()
        print("检测到旧数据库，已删除并重建。\n")

    conn = sqlite3.connect(DB_FILE)
    try:
        # 提升写入性能
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")

        print(f"共发现 {len(csv_files)} 个 CSV 文件，开始导入：\n")
        print(f"{'CSV 文件':<42}{'表名':<38}{'行数':>12}")
        print("-" * 96)

        for csv_path in csv_files:
            table = derive_table_name(csv_path)
            t0 = time.time()
            rows = import_csv_to_sqlite(csv_path, conn, table)
            create_indexes(conn, table)
            print(
                f"{csv_path.name:<42}{table:<38}{rows:>12,}   ({time.time() - t0:.1f}s)"
            )

        conn.commit()

        # 3. 校验
        verify(conn)

    finally:
        conn.close()

    size_mb = DB_FILE.stat().st_size / 1024 / 1024
    print("\n" + "=" * 60)
    print(f"完成！耗时 {time.time() - start:.1f}s，数据库大小 {size_mb:.1f} MB")
    print(f"路径：{DB_FILE}")
    print("=" * 60)
    print("\n使用示例：\n")
    print("    import sqlite3")
    print(f"    conn = sqlite3.connect(r'{DB_FILE}')")
    print("    df = pd.read_sql('SELECT * FROM orders LIMIT 5', conn)")


if __name__ == "__main__":
    main()
