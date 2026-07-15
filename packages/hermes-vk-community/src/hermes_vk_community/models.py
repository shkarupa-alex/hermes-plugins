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
    format_data: JsonObject | None = None


class KeyboardAction(VkModel):
    type: str = "text"
    label: str
    payload: str


class KeyboardButton(VkModel):
    action: KeyboardAction
    color: str = "secondary"


class VkKeyboard(VkModel):
    one_time: bool = True
    inline: bool = False
    buttons: list[list[KeyboardButton]]


class InteractionPayload(VkModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(alias="v")
    nonce: str = Field(alias="n", min_length=16, max_length=64)


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


class PhotoUploadResponse(VkModel):
    server: int
    photo: str
    hash: str


class UploadServer(VkModel):
    upload_url: str


class SavedPhoto(VkModel):
    owner_id: int
    id: int
    access_key: str | None = None


class DocumentUploadResponse(VkModel):
    file: str


class SavedDocument(VkModel):
    owner_id: int
    id: int
    access_key: str | None = None


class SaveDocumentResponse(VkModel):
    type: str | None = None
    doc: SavedDocument | None = None
    audio_message: SavedDocument | None = None


class FormattingProbeCase(VkModel):
    name: str
    operation: str
    request_message: str
    request_length: int | None = None
    request_sha256: str | None = None
    send_status: str
    readback_text: str | None = None
    readback_length: int | None = None
    readback_sha256: str | None = None
    readback_format_data: JsonObject | None = None
    error_code: int | None = None


class FormattingProbeArtifact(VkModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    api_version: str
    peer_id: str = "<redacted>"
    generated_at: str
    cases: list[FormattingProbeCase]


class CapabilityClient(VkModel):
    name: str
    version: str
    manual_visual_check: bool


class VkCapabilityProfile(VkModel):
    schema_version: int
    api_version: str
    tested_at: str
    profile: str
    rich_send: bool
    rich_edit: bool
    format_data_observed: bool
    send_limit: int
    typing_indicator_visible: bool
    observations: dict[str, str]
    clients: list[CapabilityClient]


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
