from sqlalchemy import Column, String, Text, TIMESTAMP, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

#定义历史消息的表结构

Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"

    message_id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    session_id = Column(String(16), nullable=False)
    user_question = Column(Text, nullable=False)
    model_answer = Column(Text, nullable=False)
    create_time = Column(TIMESTAMP, server_default=func.now())
    retrieval_content = Column(Text)

class KnowledgeBase(Base):
    __tablename__ = 'knowledgebases'  # 表名
    
    id = Column(Integer, primary_key=True, autoincrement=True)  # 主键
    user_id = Column(String(255), nullable=False)  # 用户 ID
    file_name = Column(String(255), nullable=False)  # 文件名称
    created_at = Column(TIMESTAMP, nullable=False, server_default='CURRENT_TIMESTAMP')  # 创建时间
    updated_at = Column(TIMESTAMP, nullable=False, server_default='CURRENT_TIMESTAMP')  # 更新时间

