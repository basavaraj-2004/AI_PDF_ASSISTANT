
import { useState } from 'react'
import './App.css'

function App() {
  // Restore the saved theme when the page loads.
  const [theme, setTheme] = useState(() => {
    try {
      const savedTheme = localStorage.getItem('theme')
      return savedTheme === 'light' || savedTheme === 'dark'
        ? savedTheme
        : 'dark'
    } catch {
      return 'dark'
    }
  })

  const [selectedFile, setSelectedFile] = useState(null)
  const [pdfName, setPdfName] = useState('')
  const [question, setQuestion] = useState('')
  const [backendStatus, setBackendStatus] = useState('')
  const [uploadStatus, setUploadStatus] = useState('')
  const [extractedText, setExtractedText] = useState('')
  const [answer, setAnswer] = useState('')
  const [isUploading, setIsUploading] = useState(false)
  const [isAsking, setIsAsking] = useState(false)
  const [isCheckingBackend, setIsCheckingBackend] = useState(false)

  // Change the theme and remember the user's choice.
  function handleThemeChange(newTheme) {
    if (newTheme !== 'light' && newTheme !== 'dark') return

    setTheme(newTheme)

    try {
      localStorage.setItem('theme', newTheme)
    } catch (error) {
      console.error('Could not save theme preference:', error)
    }
  }

  // Select and validate a PDF file.
  function handleFileChange(event) {
    const file = event.target.files?.[0]

    if (!file) return

    setAnswer('')
    setQuestion('')
    setExtractedText('')
    setUploadStatus('')
    setBackendStatus('')

    const isPdf =
      file.type === 'application/pdf' ||
      file.name.toLowerCase().endsWith('.pdf')

    if (!isPdf) {
      setSelectedFile(null)
      setPdfName('')
      setUploadStatus('Please select a valid PDF file.')
      event.target.value = ''
      return
    }

    setSelectedFile(file)
    setPdfName(file.name)
    setUploadStatus('PDF selected. Click Upload PDF to continue.')
  }

  // Upload PDF and extract its text using FastAPI.
  async function handleUpload() {
    if (!selectedFile || isUploading) return

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      setIsUploading(true)
      setExtractedText('')
      setAnswer('')
      setUploadStatus('Uploading PDF and extracting text...')

      const response = await fetch(
        'http://127.0.0.1:8000/upload',
        {
          method: 'POST',
          body: formData,
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'PDF upload failed.')
      }

      if (!data.text?.trim()) {
        setUploadStatus(
          'The PDF was uploaded, but no readable text was found. Scanned PDFs may require OCR.'
        )
        return
      }

      setExtractedText(data.text)
      setUploadStatus(
        `Upload successful! ${data.filename || selectedFile.name} is ready for questions.`
      )
    } catch (error) {
      console.error(error)
      setUploadStatus(`Upload failed: ${error.message}`)
    } finally {
      setIsUploading(false)
    }
  }

  // Check FastAPI connection.
  async function checkBackend() {
    if (isCheckingBackend) return

    try {
      setIsCheckingBackend(true)
      setBackendStatus('Checking backend connection...')

      const response = await fetch(
        'http://127.0.0.1:8000/health'
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Health check failed.')
      }

      setBackendStatus(
        data.message || 'Backend connection successful!'
      )
    } catch (error) {
      console.error(error)
      setBackendStatus(
        'Cannot connect. Make sure FastAPI and Ollama are running.'
      )
    } finally {
      setIsCheckingBackend(false)
    }
  }

  // Ask the AI about the uploaded PDF.
  async function handleAsk() {
    if (isAsking || isUploading) return

    if (!extractedText.trim()) {
      setAnswer('Please choose and upload a readable PDF first.')
      return
    }

    if (!question.trim()) {
      setAnswer('Please enter a question about your PDF.')
      return
    }

    try {
      setIsAsking(true)
      setAnswer('')

      const response = await fetch(
        'http://127.0.0.1:8000/ask',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: question.trim(),
            context: extractedText,
          }),
        }
      )

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to get an answer.')
      }

      setAnswer(data.answer || 'The AI returned no answer.')
    } catch (error) {
      console.error(error)
      setAnswer(`Unable to get an answer: ${error.message}`)
    } finally {
      setIsAsking(false)
    }
  }

  return (
    <main className={`app ${theme}-theme`}>
      <nav className="topbar" aria-label="Appearance settings">
        <div className="brand">
          <span className="brand-icon" aria-hidden="true">✦</span>
          <span>PDF Assistant</span>
        </div>

        <div className="theme-switch" aria-label="Choose color theme">
          <button
            type="button"
            className={theme === 'light' ? 'active' : ''}
            onClick={() => handleThemeChange('light')}
            aria-pressed={theme === 'light'}
          >
            ☀️ Light
          </button>

          <button
            type="button"
            className={theme === 'dark' ? 'active' : ''}
            onClick={() => handleThemeChange('dark')}
            aria-pressed={theme === 'dark'}
          >
            🌙 Dark
          </button>
        </div>
      </nav>

      <header className="header">
        <div className="eyebrow">YOUR DOCUMENT, UNDERSTOOD</div>
        <h1>AI PDF Assistant</h1>
        <p>
          Upload a document, ask anything, and find answers faster.
        </p>
      </header>

      <section className="upload-card">
        <div className="section-icon" aria-hidden="true">↥</div>

        <h2>Upload your PDF</h2>
        <p className="card-description">
          Start with a PDF document from your device.
        </p>

        <div className="upload-controls">
          <label
            htmlFor="pdf-upload"
            className="upload-button"
          >
            <span aria-hidden="true">＋</span>
            {pdfName ? 'Choose Another PDF' : 'Choose PDF'}
          </label>

          <input
            id="pdf-upload"
            className="pdf-file-input"
            type="file"
            accept=".pdf,application/pdf"
            onChange={handleFileChange}
            disabled={isUploading}
            hidden
          />

          {pdfName && (
            <div className="selected-file">
              <span className="file-icon" aria-hidden="true">PDF</span>
              <span className="selected-file-name">{pdfName}</span>
            </div>
          )}

          <button
            type="button"
            className="upload-submit"
            onClick={handleUpload}
            disabled={!selectedFile || isUploading}
          >
            {isUploading ? (
              <>
                <span className="spinner" aria-hidden="true" />
                Uploading...
              </>
            ) : (
              'Upload PDF'
            )}
          </button>
        </div>

        {uploadStatus && (
          <p
            className={`status-message ${
              uploadStatus.startsWith('Upload failed') ||
              uploadStatus.startsWith('Please') ||
              uploadStatus.startsWith('The PDF')
                ? 'status-warning'
                : ''
            }`}
            role="status"
            aria-live="polite"
          >
            {uploadStatus}
          </p>
        )}

        <div className="backend-test">
          <button
            type="button"
            onClick={checkBackend}
            disabled={isCheckingBackend || isUploading || isAsking}
          >
            {isCheckingBackend
              ? 'Checking...'
              : 'Test Backend Connection'}
          </button>

          {backendStatus && (
            <p role="status" aria-live="polite">
              {backendStatus}
            </p>
          )}
        </div>

        <p className="hint">
          Your document is ready for questions after upload.
        </p>
      </section>

      {extractedText && (
        <section className="text-preview">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">DOCUMENT CONTENT</span>
              <h2>Extracted PDF Text</h2>
            </div>

            <span className="ready-badge">● Ready</span>
          </div>

          <pre>{extractedText}</pre>
        </section>
      )}

      <section className="question-area">
        <div className="question-heading">
          <div>
            <span className="eyebrow">ASK YOUR DOCUMENT</span>
            <h2>What would you like to know?</h2>
          </div>

          <span className="question-mark" aria-hidden="true">?</span>
        </div>

        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (
              event.key === 'Enter' &&
              !event.shiftKey &&
              !isAsking &&
              !isUploading &&
              extractedText
            ) {
              event.preventDefault()
              handleAsk()
            }
          }}
          placeholder="Ask a question about your PDF..."
          rows={4}
          aria-label="Question about your PDF"
        />

        <div className="question-footer">
          <p>Press Enter to ask · Shift + Enter for a new line</p>

          <button
            type="button"
            onClick={handleAsk}
            disabled={isAsking || isUploading || !extractedText}
          >
            {isAsking ? (
              <>
                <span className="spinner" aria-hidden="true" />
                Thinking...
              </>
            ) : (
              <>
                Ask Question
                <span aria-hidden="true"> ↗</span>
              </>
            )}
          </button>
        </div>

        {answer && (
          <section className="answer-card" aria-live="polite">
            <div className="answer-heading">
              <div className="answer-icon" aria-hidden="true">✦</div>
              <div>
                <h2>AI Answer</h2>
                <p>Generated from your document</p>
              </div>
            </div>

            <p className="answer-text">{answer}</p>
          </section>
        )}
      </section>

      <footer className="footer">
        <span>AI PDF Assistant</span>
        <span>Powered by your local AI model</span>
      </footer>
    </main>
  )
}

export default App
