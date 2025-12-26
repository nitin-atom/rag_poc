# Intercom handoff plan

## Goals
- Detect when a user asks for support and prompt for name/email.
- On form submit, send name/email + full chat transcript to backend.
- Backend creates an Intercom conversation in the Inbox so agents can reply by email.

## Proposed flow
1. User submits a message.
2. Backend `/chat` response includes a `support_handoff` boolean when the message indicates a support request.
3. Frontend shows the name/email form only when `support_handoff` is true.
4. On submit, frontend calls `/support-request` with name, email, and transcript.
5. Backend calls Intercom API to create/update contact + open a conversation in the Inbox with the transcript.

## Backend changes (FastAPI)
- Add `POST /support-request` endpoint.
  - Request body: `name`, `email`, `transcript` (array of messages with role + content), optional `last_user_message`.
  - Response: success status + intercom conversation id (if available).
- Add support detection to `/chat` flow.
  - Use a lightweight LLM classifier call (OpenAI) that returns `support_handoff` boolean.
  - Return `support_handoff` in `ChatResponse` alongside `answer` and `citations`.
- Intercom integration logic (inside `/support-request`):
  - Read `INTERCOM_ACCESS_TOKEN` and `INTERCOM_WORKSPACE_ID` from env.
  - Create or update a contact by email.
  - Create a conversation in the Inbox, using the transcript as the initial message content.
  - Use conversation subject: `AI chatbot`.
  - Include metadata: name, email, timestamp, and transcript.
  - Handle Intercom errors with clear HTTP 4xx/5xx messages.

## Frontend changes (Next.js)
- Track `supportHandoff` in state based on `/chat` response.
- When `supportHandoff` becomes true:
  - Render a name/email form below the chat.
- On form submit:
  - Send `name`, `email`, and `messages` transcript to `/support-request`.
  - Show confirmation text like “We will be reaching out to you over email.” and end that conversation.
  - Allow user to start a new chat via existing “New Chat” action.
  - Handle errors with a user‑friendly message.

## Data contract
- `ChatResponse` adds `support_handoff: boolean`.
- `SupportRequest` payload:
  - `name: string`
  - `email: string`
  - `transcript: [{ role: 'user'|'assistant', content: string }]`
  - Optional: `source: 'web'`, `session_id` if needed later.

## Configuration
- Add to `.env` (backend):
  - `INTERCOM_ACCESS_TOKEN`
  - `INTERCOM_WORKSPACE_ID`
- Ensure CORS includes frontend origin for `/support-request`.

## Error handling & observability
- Validate name/email/transcript on the backend.
- Return clear errors for Intercom failures (and log status/body).
- Frontend shows “We’ve received your request” on success, and a retry option on failure.

## Testing (manual)
- Trigger support request in chat and verify form appears.
- Submit form and verify Intercom conversation appears in Inbox.
- Confirm transcript appears in the Intercom message.

## Open questions
- LLM classifier for support intent.
- Conversation subject/label: `AI chatbot`.
