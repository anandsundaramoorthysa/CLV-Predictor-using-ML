# home.html - Home Page Template Documentation

## Overview
`home.html` is the main user interface template that handles file uploads, displays real-time processing progress, and shows business insights after analysis completion.

## Table of Contents
1. [Page Structure](#page-structure)
2. [File Upload Section](#file-upload-section)
3. [Progress Tracker Section](#progress-tracker-section)
4. [Business Insights Section](#business-insights-section)
5. [JavaScript Functionality](#javascript-functionality)
6. [User Interaction Flow](#user-interaction-flow)

---

## Page Structure

### HTML Layout

```html
<!DOCTYPE html>
<html>
  <head>
    - Meta tags
    - CSS stylesheet link
  </head>
  <body>
    <div class="container">
      - Header
      - Flash Messages
      - File Upload Section
      - Progress Tracker Section (hidden initially)
      - Business Insights Section (conditional)
    </div>
    <script>
      - JavaScript functionality
    </script>
  </body>
</html>
```

---

## File Upload Section

### Purpose
Allows users to upload CSV or Excel files for analysis.

### Components

#### Upload Form
```html
<form method="POST" enctype="multipart/form-data" id="uploadForm">
  - File input (accepts .csv, .xlsx, .xls)
  - Submit button ("Upload & Analyze")
</form>
```

#### File Input Wrapper
- Custom styled file input
- Visual feedback on file selection
- Icon changes: 📁 → ✅ when file selected
- Border color changes on selection

### Features
- **File Type Validation**: Only accepts CSV/Excel files
- **Visual Feedback**: Shows selected filename
- **Required Field**: Prevents empty submissions

---

## Progress Tracker Section

### Purpose
Displays real-time processing status with step-by-step updates.

### Structure

#### Progress Section Container
- Hidden by default (`display: none`)
- Shown when processing starts
- Contains progress tracker and error messages

#### Progress Steps (11 Steps)
1. File uploaded
2. Data validation
3. Data preprocessing
4. Feature engineering
5. Train model 1 – Linear Regression
6. Train model 2 – Random Forest
7. Train model 3 – Gradient Boosting
8. Model evaluation
9. Best model selection
10. Final prediction generation
11. Business insights ready

### Visual Indicators

#### Status Icons
- **⏸ Pending**: Step not started (gray)
- **⏳ In Progress**: Step currently running (yellow, animated)
- **✅ Completed**: Step finished (green)

#### CSS Classes
- `.progress-step`: Base step container
- `.in-progress`: Active step styling (yellow border, pulse animation)
- `.completed`: Finished step styling (green border)

### Error Display
- Error message container
- Shows user-friendly error messages
- Re-enables form on error

---

## Business Insights Section

### Purpose
Displays five business insights after processing completes.

### Insight Components

#### 1. Customer Segmentation
**Display Elements**:
- Number of segments
- Segment size table
- Percentage distribution
- Segment interpretations
- Scatter plot visualization

**Data Structure**:
```javascript
{
  segment: "Segment 1",
  size: 150,
  percentage: 50.0,
  interpretation: "Customer group description"
}
```

#### 2. Customer Lifetime Value (CLV) Prediction
**Display Elements**:
- CLV statistics (mean, median, std)
- Distribution histogram
- Top high-value customers table

**Data Structure**:
```javascript
{
  distribution: {
    mean: 1000.50,
    median: 950.00,
    std: 200.30
  },
  top_customers: [
    {clv: 1500.00},
    ...
  ]
}
```

#### 3. Churn Prediction
**Display Elements**:
- Risk distribution counts
- High-risk customers table
- Churn probability values

**Data Structure**:
```javascript
{
  risk_counts: {
    High: 50,
    Medium: 100,
    Low: 150
  },
  high_risk_customers: [
    {churn_probability: 0.85},
    ...
  ]
}
```

#### 4. Market Basket Analysis
**Display Elements**:
- Top product combinations table
- Support values
- Confidence values

**Data Structure**:
```javascript
{
  combination: "Product A → Product B",
  support: 0.15,
  confidence: 0.75
}
```

#### 5. Sales/Demand Forecasting
**Display Elements**:
- Actual vs Predicted trend chart
- Forecast table with periods

**Data Structure**:
```javascript
{
  period: "Period 1",
  predicted: 1250.50
}
```

---

## JavaScript Functionality

### Initialization

#### Session Check
```javascript
const sessionId = {{ session_id|tojson if session_id else 'null' }};
```
- Gets session ID from Flask template
- Used for progress tracking

#### Progress Polling
- Starts automatically if session exists
- Checks if processing is active
- Initializes UI accordingly

### Core Functions

#### `initializeProgressTracker()`
**Purpose**: Creates progress step elements dynamically

**Process**:
1. Shows progress section
2. Creates 11 step divs
3. Sets initial status (⏸ pending)
4. Adds to DOM

**Step Structure**:
```html
<div class="progress-step" id="step-1">
  <span class="step-icon" id="icon-1">⏸</span>
  <span class="step-name">File uploaded</span>
</div>
```

#### `startProgressPolling()`
**Purpose**: Polls `/progress` endpoint for updates

**Process**:
1. Sets interval (1 second)
2. Fetches progress from API
3. Updates UI with new status
4. Stops when completed or error

**Polling Logic**:
```javascript
setInterval(async () => {
  const response = await fetch('/progress');
  const data = await response.json();
  updateProgressUI(data);
  
  if (data.completed || data.error) {
    stopProgressPolling();
  }
}, 1000);
```

#### `updateProgressUI(data)`
**Purpose**: Updates visual status of each step

**Process**:
1. Checks for errors
2. Updates each step's icon and class
3. Disables/enables form based on status
4. Handles completion state

**Status Updates**:
- `pending` → ⏸ (gray)
- `in_progress` → ⏳ (yellow, animated)
- `completed` → ✅ (green)

#### `checkProgressStatus()`
**Purpose**: Checks if processing is active on page load

**Process**:
1. Fetches current progress
2. Initializes tracker if active
3. Starts polling if needed
4. Disables form if processing

#### `disableForm(disable)`
**Purpose**: Enables/disables form during processing

**Process**:
- Adds/removes `processing` class
- Disables/enables submit button
- Disables/enables file input

**Visual Effects**:
- Opacity reduction when disabled
- Cursor change to not-allowed
- Prevents multiple submissions

### Form Submission Handler

#### Event Listener
```javascript
uploadForm.addEventListener('submit', function(e) {
  // Check if already processing
  if (progressInterval) {
    e.preventDefault();
    alert('Processing already in progress');
    return false;
  }
  
  // Show progress section
  progressSection.style.display = 'block';
  initializeProgressTracker();
  
  // Allow form submission
  return true;
});
```

### File Selection Feedback

#### Visual Updates
- **On Selection**:
  - Icon: 📁 → ✅
  - Border: Blue → Green
  - Background: White → Light green
  - Text: Shows filename

- **On Clear**:
  - Reverts to default state

---

## User Interaction Flow

### Complete User Journey

```
1. User visits page
   ↓
2. Sees upload form
   ↓
3. Selects CSV/Excel file
   ↓
4. Visual feedback (icon changes)
   ↓
5. Clicks "Upload & Analyze"
   ↓
6. Form submits to backend
   ↓
7. Progress section appears
   ↓
8. Progress tracker initializes
   ↓
9. JavaScript starts polling /progress
   ↓
10. Steps update in real-time:
    - ⏸ → ⏳ → ✅
   ↓
11. Form disabled during processing
   ↓
12. All steps complete
   ↓
13. Page reloads automatically
   ↓
14. Business insights displayed
   ↓
15. Form re-enabled
   ↓
16. User can view insights or upload new file
```

### Error Handling Flow

```
1. Error occurs in processing
   ↓
2. Error message set in progress state
   ↓
3. Frontend receives error in poll
   ↓
4. Error message displayed
   ↓
5. Form re-enabled
   ↓
6. User can retry upload
```

---

## CSS Styling

### Progress Step States

#### Pending State
```css
.progress-step {
  border-left: 4px solid #dee2e6; /* Gray */
  background: white;
}
```

#### In Progress State
```css
.progress-step.in-progress {
  border-left-color: #ffc107; /* Yellow */
  background: #fffbf0;
  animation: pulse 2s infinite;
}
```

#### Completed State
```css
.progress-step.completed {
  border-left-color: #28a745; /* Green */
  background: #f0f9f4;
}
```

### Animation
```css
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.8; }
}
```

---

## Data Flow

### Template Variables

#### From Flask Backend
```python
render_template('home.html', 
  insights=insights,      # Business insights dict
  session_id=session_id   # Current session ID
)
```

#### Insights Structure
```python
insights = {
  'segmentation': segment_info,  # List of segments
  'clv': clv_info,               # CLV stats and top customers
  'churn': churn_info,           # Risk counts and high-risk
  'basket': basket_info,         # Product combinations
  'forecast': forecast_info      # Forecast periods
}
```

---

## Key Features

### 1. Real-Time Updates
- Polls every 1 second
- Updates UI without page refresh
- Smooth visual transitions

### 2. User Feedback
- Clear visual indicators
- Progress percentage implied
- Error messages displayed

### 3. Form Management
- Prevents duplicate submissions
- Disables during processing
- Re-enables on completion/error

### 4. Responsive Design
- Works on different screen sizes
- Clean, modern UI
- Professional appearance

---

## Browser Compatibility

### Required Features
- **Fetch API**: For progress polling
- **ES6+ JavaScript**: Arrow functions, async/await
- **CSS Animations**: For progress indicators

### Fallbacks
- Error handling for failed requests
- Graceful degradation if JavaScript disabled

---

## Important Notes

1. **Session Management**: Relies on Flask session for state (optimized - no large data)
2. **Polling Frequency**: 1 second (adjustable)
3. **Auto-Reload**: Page reloads when processing completes
4. **Error Recovery**: Form re-enables on errors
5. **Visual Feedback**: All user actions have visual responses
6. **Progress Tracking**: Real-time updates via `/progress` endpoint
7. **Error Display**: User-friendly error messages shown in progress section
8. **Form Management**: Prevents duplicate submissions during processing

---

## Recent Improvements

### Progress Tracking
- Real-time step-by-step progress updates
- Visual indicators (⏳ in-progress, ✅ completed)
- Error messages displayed in progress section
- Automatic form disabling during processing

### Error Handling
- Graceful error display in progress tracker
- Form re-enables on errors
- User-friendly error messages

### Session Optimization
- Large DataFrames not stored in session
- Prevents browser header size errors
- Only metadata stored in Flask session

## Future Enhancements

- WebSocket support for real-time updates (no polling)
- Progress percentage display
- Estimated time remaining
- Cancel processing option
- Download results as PDF/Excel
- Share results via URL
- Export full dataset results

