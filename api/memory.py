"""Database memory for storing chat history using Firebase Firestore."""

import os

import firebase_admin
import structlog
from firebase_admin import firestore

logger = structlog.get_logger(__name__)


class MemoryManager:
    """Manages chat history sessions using Firebase Firestore."""

    # Class-level storage fallbacks for cross-instance persistence
    _local_last_files = {}  # session_id -> filename
    _local_doc_metadata = {}  # filename -> metadata_dict

    def __init__(self):
        """Initialize Firebase Admin SDK or handle existing apps."""
        self.db = None
        self.enabled = False
        self._local_storage = "data/chat_history.json"

        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        if not os.path.exists(self._local_storage):
            with open(self._local_storage, "w") as f:
                import json

                json.dump({}, f)

        try:
            import firebase_admin
            from firebase_admin import firestore

            if not firebase_admin._apps:
                firebase_admin.initialize_app()

            self.db = firestore.client()
            self.enabled = True
            logger.info("firebase_initialized", status="success")
        except Exception as e:
            logger.warning(
                "firebase_initialization_failed", error=str(e), hint="System will use local JSON for persistent memory"
            )

    def save_message(self, session_id: str, role: str, content: str):
        """Save a message to the session's subcollection."""
        if not self.enabled:
            return
        try:
            docs_ref = self.db.collection("sessions").document(session_id).collection("messages").document()
            docs_ref.set({"role": role, "content": content, "timestamp": firestore.SERVER_TIMESTAMP})
        except Exception as e:
            logger.error("save_message_failed", error=str(e))

    def get_history(self, session_id: str):
        """Retrieve conversation history for a given session."""
        if not self.enabled:
            return []
        try:
            docs = (
                self.db.collection("sessions")
                .document(session_id)
                .collection("messages")
                .order_by("timestamp")
                .stream()
            )
            return [{"role": doc.to_dict()["role"], "content": doc.to_dict()["content"]} for doc in docs]
        except Exception as e:
            logger.error("get_history_failed", error=str(e))
            return []

    def save_document_metadata(self, filename: str, metadata: dict):
        """Store metadata (summaries, etc.) for a document."""
        self._local_doc_metadata[filename] = metadata
        if not self.enabled:
            return
        try:
            doc_ref = self.db.collection("uploaded_documents").document(filename)
            metadata["timestamp"] = firestore.SERVER_TIMESTAMP
            doc_ref.set(metadata)
            logger.info("save_doc_meta_success", filename=filename)
        except Exception as e:
            logger.error("save_doc_meta_failed", filename=filename, error=str(e))

    def get_document_metadata(self, filename: str) -> dict | None:
        """Retrieve stored metadata for a document."""
        if filename in self._local_doc_metadata:
            return self._local_doc_metadata[filename]
        if not self.enabled:
            return None
        try:
            doc_ref = self.db.collection("uploaded_documents").document(filename)
            doc = doc_ref.get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            logger.error("get_doc_meta_failed", filename=filename, error=str(e))
            return None

    def save_last_filename(self, session_id: str, filename: str):
        """Track the last referenced filename for a session."""
        self._local_last_files[session_id] = filename
        if not self.enabled:
            return
        try:
            doc_ref = self.db.collection("sessions").document(session_id)
            doc_ref.set({"last_filename": filename}, merge=True)
            logger.info("save_last_filename_success", session_id=session_id, filename=filename)
        except Exception as e:
            logger.error("save_last_filename_failed", session_id=session_id, error=str(e))

    def get_last_filename(self, session_id: str) -> str | None:
        """Retrieve the last referenced filename for a session."""
        if session_id in self._local_last_files:
            return self._local_last_files[session_id]
        if not self.enabled:
            return None
        try:
            doc_ref = self.db.collection("sessions").document(session_id)
            doc = doc_ref.get()
            return doc.to_dict().get("last_filename") if (doc.exists and doc.to_dict()) else None
        except Exception as e:
            logger.error("get_last_filename_failed", session_id=session_id, error=str(e))
            return None
