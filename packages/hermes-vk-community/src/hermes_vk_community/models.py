from __future__ import annotations
from typing import Generic, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, JsonValue

T = TypeVar("T")
JsonObject: TypeAlias = dict[str, JsonValue]


class VkModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class VkApiError(VkModel):
    error_code: int
    error_msg: str
    request_params: list[JsonObject] = Field(default_factory=list[JsonObject])


class VkApiEnvelope(VkModel, Generic[T]):
    response: T | None = None
    error: VkApiError | None = None


class LongPollLease(VkModel):
    key: str
    server: str
    ts: str


class LongPollResponse(VkModel):
    ts: str | None = None
    updates: list[JsonObject] = Field(default_factory=list[JsonObject])
    failed: int | None = None


class CommunityLongPollEvents(VkModel):
    message_new: int = 0


class CommunityLongPollSettings(VkModel):
    is_enabled: bool = False
    api_version: str = ""
    events: CommunityLongPollEvents = Field(default_factory=CommunityLongPollEvents)


class PhotoSize(VkModel):
    url: str
    width: int = 0
    height: int = 0


class PhotoAttachment(VkModel):
    sizes: list[PhotoSize] = Field(default_factory=list[PhotoSize])


class DocumentAttachment(VkModel):
    url: str | None = None
    title: str = "document"
    ext: str = ""


class AudioMessageAttachment(VkModel):
    link_ogg: str | None = None
    link_mp3: str | None = None
    duration: int = 0


class AudioAttachment(VkModel):
    url: str | None = None
    title: str = "audio"


class VkAttachment(VkModel):
    type: str
    photo: PhotoAttachment | None = None
    doc: DocumentAttachment | None = None
    audio_message: AudioMessageAttachment | None = None
    audio: AudioAttachment | None = None
    video: JsonObject | None = None
    sticker: JsonObject | None = None
    link: JsonObject | None = None


class VkMessage(VkModel):
    id: int
    conversation_message_id: int | None = None
    date: int
    peer_id: int
    from_id: int
    text: str = ""
    random_id: int = 0
    attachments: list[VkAttachment] = Field(default_factory=list[VkAttachment])
    reply_message: JsonObject | None = None
    fwd_messages: list[JsonObject] = Field(default_factory=list[JsonObject])
    payload: str | None = None


class MessageNewObject(VkModel):
    message: VkMessage
    client_info: JsonObject | None = None


class VkUpdate(VkModel):
    type: str
    object: MessageNewObject | JsonObject
    group_id: int
    event_id: str | None = None


class SendResponse(VkModel):
    message_id: int


class User(VkModel):
    id: int
    first_name: str = ""
    last_name: str = ""


class Group(VkModel):
    id: int
    name: str = ""
    type: str = ""
    is_closed: int | None = None


class GroupsResponse(VkModel):
    groups: list[Group] = Field(default_factory=list[Group])
