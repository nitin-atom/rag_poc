'use client'

import { useState, useRef, useEffect } from 'react'

interface Citation {
  title: string
  url: string
}

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  citations?: Citation[]
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isFocused, setIsFocused] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e: React.FormEvent, starterQuestion?: string) => {
    e.preventDefault()
    const question = starterQuestion?.trim() || input.trim()
    if (!question || isLoading) return
    setInput('')

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: question
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      })

      if (!response.ok) throw new Error('Failed to get response')

      const data = await response.json()

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: data.answer,
        citations: data.citations
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleNewChat = () => {
    setMessages([])
    setInput('')
    setIsLoading(false)
  }

  const handleStarter = (question: string) => {
    handleSubmit({ preventDefault: () => {} } as React.FormEvent, question)
  }

  const starters = [
    'How do I sell a domain?',
    'How do I change my name servers?',
    'How do I transfer a domain?',
  ]

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div style={styles.headerInner}>
          <div style={styles.logo}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 16v-4" />
              <path d="M12 8h.01" />
            </svg>
            <span style={styles.logoText}>Atom Help</span>
          </div>
          <button
            onClick={handleNewChat}
            style={styles.newChatButton}
            title="New Chat"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 5v14" />
              <path d="M5 12h14" />
            </svg>
            <span>New Chat</span>
          </button>
        </div>
      </header>

      <main style={styles.main}>
        <div style={styles.messagesContainer}>
          {messages.length === 0 ? (
            <div style={styles.emptyState}>
              <div style={styles.emptyIcon}>
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
              </div>
              <h2 style={styles.emptyTitle}>Ask about Atom</h2>
              <p style={styles.emptyText}>
                Get answers about buying or selling domains, updating name servers, getting paid and more.
              </p>
              <div style={styles.starters}>
                {starters.map((starter, i) => (
                  <button
                    key={i}
                    onClick={() => handleStarter(starter)}
                    style={styles.starterButton}
                  >
                    {starter}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div style={styles.messages}>
              {messages.map((message, index) => (
                <div
                  key={message.id}
                  style={{
                    ...styles.messageWrapper,
                    justifyContent: message.type === 'user' ? 'flex-end' : 'flex-start',
                    animation: 'fadeInUp 0.3s ease-out forwards',
                    animationDelay: `${index * 0.05}s`
                  }}
                >
                  <div
                    style={{
                      ...styles.message,
                      ...(message.type === 'user' ? styles.userMessage : styles.assistantMessage)
                    }}
                  >
                    <p style={styles.messageContent}>{message.content}</p>
                    {message.citations && message.citations.length > 0 && (
                      <div style={styles.citations}>
                        <span style={styles.citationLabel}>Sources:</span>
                        {message.citations.map((citation, i) => (
                          <a
                            key={i}
                            href={citation.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            style={styles.citationLink}
                          >
                            {citation.title}
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginLeft: 4 }}>
                              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                              <polyline points="15 3 21 3 21 9" />
                              <line x1="10" y1="14" x2="21" y2="3" />
                            </svg>
                          </a>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div style={{ ...styles.messageWrapper, justifyContent: 'flex-start', opacity: 1 }}>
                  <div style={{ ...styles.message, ...styles.assistantMessage, ...styles.loadingMessage }}>
                    <div style={styles.loadingCursor} />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </main>

      <footer style={styles.footer}>
        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={{
            ...styles.inputWrapper,
            borderColor: isFocused ? 'var(--text-tertiary)' : 'var(--border)',
          }}>
            <textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="Ask a question..."
              style={styles.input}
              rows={1}
              disabled={isLoading}
            />
            <button
              type="submit"
              style={{
                ...styles.sendButton,
                opacity: input.trim() && !isLoading ? 1 : 0.4,
                cursor: input.trim() && !isLoading ? 'pointer' : 'not-allowed'
              }}
              disabled={!input.trim() || isLoading}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
        </form>
      </footer>
    </div>
  )
}

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    maxWidth: 720,
    margin: '0 auto',
    background: 'var(--bg-secondary)',
    borderLeft: '1px solid var(--border)',
    borderRight: '1px solid var(--border)',
  },
  header: {
    borderBottom: '1px solid var(--border)',
    background: 'var(--bg-secondary)',
  },
  headerInner: {
    padding: '16px 24px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  logo: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    color: 'var(--text-primary)',
  },
  logoText: {
    fontSize: 15,
    fontWeight: 600,
    letterSpacing: '-0.01em',
  },
  newChatButton: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '8px 12px',
    background: 'var(--bg-tertiary)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    fontSize: 13,
    fontWeight: 500,
    color: 'var(--text-secondary)',
    cursor: 'pointer',
    transition: 'all 0.15s ease',
    fontFamily: 'inherit',
  },
  main: {
    flex: 1,
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },
  messagesContainer: {
    flex: 1,
    overflow: 'auto',
    padding: '24px',
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    textAlign: 'center',
    padding: '40px 24px',
  },
  emptyIcon: {
    color: 'var(--text-tertiary)',
    marginBottom: 16,
  },
  emptyTitle: {
    fontSize: 20,
    fontWeight: 600,
    color: 'var(--text-primary)',
    marginBottom: 8,
    letterSpacing: '-0.02em',
  },
  emptyText: {
    fontSize: 14,
    color: 'var(--text-secondary)',
    maxWidth: 320,
  },
  starters: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 24,
    justifyContent: 'center',
  },
  starterButton: {
    padding: '8px 14px',
    fontSize: 13,
    color: 'var(--text-secondary)',
    background: 'var(--bg-tertiary)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    cursor: 'pointer',
    transition: 'all 0.15s ease',
    fontFamily: 'inherit',
  },
  messages: {
    display: 'flex',
    flexDirection: 'column',
    gap: 16,
  },
  messageWrapper: {
    display: 'flex',
    opacity: 0,
  },
  message: {
    maxWidth: '85%',
    padding: '12px 16px',
    borderRadius: 'var(--radius)',
  },
  userMessage: {
    background: 'var(--text-primary)',
    color: 'white',
    borderBottomRightRadius: 4,
  },
  assistantMessage: {
    background: 'var(--bg-tertiary)',
    border: '1px solid var(--border)',
    borderBottomLeftRadius: 4,
  },
  messageContent: {
    fontSize: 14,
    lineHeight: 1.6,
    whiteSpace: 'pre-wrap',
  },
  citations: {
    marginTop: 12,
    paddingTop: 12,
    borderTop: '1px solid var(--border)',
    display: 'flex',
    flexDirection: 'column',
    gap: 6,
  },
  citationLabel: {
    fontSize: 11,
    fontWeight: 500,
    color: 'var(--text-tertiary)',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
    marginBottom: 4,
  },
  citationLink: {
    fontSize: 13,
    color: 'var(--accent)',
    textDecoration: 'none',
    display: 'inline-flex',
    alignItems: 'center',
    transition: 'opacity 0.15s ease',
  },
  loadingMessage: {
    minWidth: 60,
    minHeight: 24,
    display: 'flex',
    alignItems: 'center',
    padding: '14px 16px',
  },
  loadingCursor: {
    width: 3,
    height: 20,
    background: 'var(--text-secondary)',
    borderRadius: 1,
    animation: 'blink 1s ease-in-out infinite',
  },
  footer: {
    borderTop: '1px solid var(--border)',
    padding: '16px 24px',
    background: 'var(--bg-secondary)',
  },
  form: {
    display: 'flex',
    gap: 12,
  },
  inputWrapper: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    padding: '10px 12px 10px 16px',
    background: 'var(--bg-tertiary)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius)',
    transition: 'border-color 0.15s ease, box-shadow 0.15s ease',
  },
  input: {
    flex: 1,
    border: 'none',
    background: 'transparent',
    fontSize: 14,
    color: 'var(--text-primary)',
    outline: 'none',
    resize: 'none',
    fontFamily: 'inherit',
    lineHeight: 1.5,
  },
  sendButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: 36,
    height: 36,
    borderRadius: 'var(--radius-sm)',
    background: 'var(--text-primary)',
    color: 'white',
    border: 'none',
    transition: 'opacity 0.15s ease, transform 0.15s ease',
    flexShrink: 0,
  },
}
