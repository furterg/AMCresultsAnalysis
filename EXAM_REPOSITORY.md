# Exam Repository - Historical Tracking Guide

The **Exam Repository** feature allows you to automatically store exam metrics over time for comparative analysis and historical tracking. This enables you to:

- Track exam performance trends over multiple sessions
- Compare similar exams (same course, different cohorts)
- Build institutional knowledge about question quality
- Analyze long-term patterns in student performance
- Monitor exam difficulty consistency across time

## 📊 What Gets Stored

For each exam analysis, the following metrics are automatically saved:

### Exam Identification
- **project_name**: Unique identifier (e.g., `202511-MATH101_Final`)
- **analysis_date**: When the analysis was run
- **year**, **month**, **course_code**: Auto-parsed from project name

### Student Metrics
- **num_students**: Number of examinees (integer)
- **num_questions**: Number of questions (integer)

### Grade Statistics
- **avg_grade**: Mean score (2 decimals)
- **median_grade**: Median score (2 decimals)
- **std_dev_grade**: Standard deviation (2 decimals)
- **min_grade**: Minimum achieved score (2 decimals)
- **max_grade**: Maximum achieved score (2 decimals)

### Psychometric Metrics (Classical Test Theory)
- **avg_difficulty**: Average difficulty index (3 decimals)
- **avg_discrimination**: Average discrimination index (3 decimals)
- **avg_correlation**: Average point-biserial correlation (3 decimals)
- **pass_rate**: Percentage of students passing (3 decimals)
- **cronbach_alpha**: Reliability coefficient (3 decimals)

---

## 🚀 Quick Start

### Option 1: Airtable (Recommended)

**Why Airtable?**
- Free tier (1,000 records)
- Beautiful web interface
- Built-in charts and views
- Mobile app available
- No technical setup required

**Setup (5 minutes):**

1. **Create Airtable Account**
   - Go to https://airtable.com and sign up for free

2. **Create a New Base**
   - Click "Start from scratch"
   - Name it: "AMC Exam Repository"

3. **Get Your API Key**
   - Click your profile → Developer Hub → Personal Access Tokens
   - Click "Create new token"
   - Give it a name: "AMC Report Generator"
   - Add these scopes:
     - `data.records:read` - Read existing records
     - `data.records:write` - Save exam metrics
     - `schema.bases:read` - Check table structure
     - `schema.bases:write` - Auto-create table/fields ⭐
   - Grant access to your "AMC Exam Repository" base
   - Copy the token (starts with `pat...`)

4. **Get Your Base ID**
   - Open your base
   - Click "Help" → "API documentation"
   - Find "The ID of this base is..." (starts with `app...`)

5. **Add to `.env` File**
   ```bash
   AMC_REPOSITORY_BACKEND=airtable
   AIRTABLE_API_KEY=patXXXXXXXXXXXXXX
   AIRTABLE_BASE_ID=appXXXXXXXXXXXXXX
   ```

6. **Run Your First Analysis**
   - The script will automatically create the "Exams" table with all fields!
   - Just confirm when prompted

**That's it!** Every report you generate will now save metrics to Airtable.

---

### Option 2: Baserow (Open Source Alternative)

**Why Baserow?**
- Fully open source
- Can self-host
- Similar to Airtable
- Good for data privacy requirements

**Setup:**

1. **Create Baserow Account**
   - Cloud: https://baserow.io (free tier)
   - Self-hosted: Follow [Baserow docs](https://baserow.io/docs/installation%2Finstall-with-docker)

2. **Create Database & Table**
   - Create a new database
   - Create a table named "Exams"

3. **Get API Credentials**
   - Profile → API tokens → Create new token
   - Copy the token
   - Get Database ID from URL: `https://baserow.io/database/[ID]`
   - Get Table ID from URL when viewing table

4. **Add to `.env` File**
   ```bash
   AMC_REPOSITORY_BACKEND=baserow
   BASEROW_API_KEY=your_api_key_here
   BASEROW_DATABASE_ID=12345
   BASEROW_TABLE_ID=67890
   ```

---

## 🔧 Configuration Reference

### Environment Variables

You can set these in your `.env` file or as system environment variables:

```bash
# Repository Backend (required to enable feature)
AMC_REPOSITORY_BACKEND=airtable  # Options: 'airtable', 'baserow', 'none'

# Airtable Configuration
AIRTABLE_API_KEY=patXXXXXXXXXX         # Your Personal Access Token
AIRTABLE_BASE_ID=appXXXXXXXXXX         # Your Base ID
AIRTABLE_TABLE_NAME=Exams              # Optional, defaults to 'Exams'

# Baserow Configuration
BASEROW_API_KEY=your_token_here        # Your API token
BASEROW_DATABASE_ID=12345              # Numeric Database ID
BASEROW_TABLE_ID=67890                 # Numeric Table ID
```

### Alternative Variable Names

For compatibility, you can also use these prefixed versions:

```bash
AMC_REPOSITORY_BACKEND=airtable
AMC_AIRTABLE_API_KEY=patXXXXXXXXXX
AMC_AIRTABLE_BASE_ID=appXXXXXXXXXX
```

---

## 💡 Usage

### Automatic Saving After Report Generation

Once configured, the repository is integrated into your normal workflow:

1. **Run Analysis as Usual**
   ```bash
   python amcreport.py
   ```

2. **After PDF Generation**
   - You'll see a prompt:
     ```
     ============================================================
     Exam Repository: Save Metrics
     ============================================================
     Save exam metrics to repository? [Y/n]:
     ```

3. **Confirm to Save**
   - Press `Y` or just `Enter` to save
   - Press `n` to skip this time

4. **Success Message**
   ```
   ✓ Exam metrics saved successfully to airtable
   ```

### UPSERT Behavior

The system uses **UPSERT** logic (Update or Insert):

- **Same project name?** → Updates the existing record
- **New project name?** → Creates a new record

This means you can re-run analyses (e.g., after correcting grades) and the metrics will be updated automatically.

### Project Name Format

For best results, use this naming format: `YYYYMM-CourseCode`

**Examples:**
- `202511-MATH101_Final` → Year: 2025, Month: 11, Course: MATH101_Final
- `202403-BIO205_Midterm` → Year: 2024, Month: 03, Course: BIO205_Midterm

This allows filtering/grouping by:
- Academic year
- Semester/month
- Course code

---

## 📈 Analyzing Your Data

### In Airtable

**Create Views:**
1. **By Course**: Group by `course_code`
2. **By Time**: Group by `year` + `month`
3. **Recent Exams**: Filter: `analysis_date > 30 days ago`
4. **Low Performance**: Filter: `avg_grade < 12`

**Create Charts:**
1. Click "Create" → "Chart"
2. Example: Grade trends over time
   - X-axis: `analysis_date`
   - Y-axis: `avg_grade`
   - Group by: `course_code`

**Export Data:**
- Click "..." → "Download CSV"
- Use in Excel, R, Python, etc.

### In Baserow

Similar functionality:
- Create filtered views
- Export to CSV
- Use Baserow's API for custom analytics

---

## 🔍 Example Use Cases

### 1. Track Exam Difficulty Over Time

**Question:** "Are my exams getting harder or easier?"

**Analysis:**
- Filter by course code: `MATH101`
- Sort by `analysis_date`
- Compare `avg_difficulty` over time

### 2. Compare Cohorts

**Question:** "How does this year's class compare to last year?"

**Analysis:**
- Filter: `course_code = MATH101_Final`
- Compare rows:
  - `202411-MATH101_Final` (this year)
  - `202311-MATH101_Final` (last year)
- Look at: `avg_grade`, `pass_rate`, `avg_discrimination`

### 3. Identify Problematic Exams

**Question:** "Which exams have poor question quality?"

**Analysis:**
- Filter: `avg_discrimination < 0.2`
- Or: `avg_correlation < 0.15`
- Review those exams for question improvements

### 4. Monitor Reliability

**Question:** "Are my exams internally consistent?"

**Analysis:**
- Check `cronbach_alpha` values
- Good: α > 0.70
- Questionable: α = 0.60-0.69
- Poor: α < 0.60

---

## 🛠️ Troubleshooting

### "Failed to initialize repository backend"

**Possible causes:**
1. Missing or invalid API key
2. Missing Base/Database ID
3. Network connectivity issues

**Solutions:**
- Verify your `.env` file has the correct values
- Test your API key in Airtable/Baserow web interface
- Check internet connection

### "Failed to set up Airtable table"

**Cause:** Your Personal Access Token doesn't have `schema.bases:write` scope

**Solution:**
1. Go to Airtable → Developer Hub → Your token
2. Edit scopes and add `schema.bases:write`
3. Re-run the script

### "Table exists but fields are missing"

**What happened:** You manually created the table but it's missing fields

**Solution:**
- Let the script add the missing fields (just accept the prompt)
- Or delete the table and let the script recreate it

### Data Not Saving

**Check:**
1. Is `AMC_REPOSITORY_BACKEND` set? (must be 'airtable' or 'baserow')
2. Are credentials correct?
3. Do you see error messages in the log file?

**Debug:**
```bash
# Check if repository is enabled
python -c "from settings import get_settings; s = get_settings(); print(f'Backend: {s.repository_backend}')"
```

---

## 🔒 Privacy & Security

### What Data is Stored

**Stored:**
- Aggregated statistics (averages, counts)
- Project names and dates
- Psychometric indices

**NOT Stored:**
- Individual student names
- Student IDs
- Specific answer data
- Any personally identifiable information (PII)

### API Key Security

**Best Practices:**
- Never commit `.env` files to version control
- Use environment variables on production systems
- Rotate API keys periodically
- Use read-only tokens where write access isn't needed

### Data Retention

**Airtable:**
- Free tier: 1,000 records (plenty for years of exams)
- Paid tier: Unlimited
- You control deletion

**Baserow:**
- Self-hosted: You control everything
- Cloud: Check their retention policies

---

## ⚙️ Advanced Configuration

### Disable Repository Temporarily

Set in `.env`:
```bash
AMC_REPOSITORY_BACKEND=none
```

Or simply don't set it (defaults to 'none').

### Use Different Table Name

```bash
AIRTABLE_TABLE_NAME=MyExamsTable
```

### Custom Pass Threshold

By default, pass rate uses 10.0 as the threshold. This is calculated in the code but not currently configurable. Contact the maintainer if you need this feature.

---

## 📦 Data Schema Reference

For developers or advanced users who want to work with the data directly:

```json
{
  "project_name": "string (unique key)",
  "analysis_date": "date (YYYY-MM-DD)",
  "year": "string (YYYY)",
  "month": "string (MM)",
  "course_code": "string",
  "num_students": "integer",
  "num_questions": "integer",
  "avg_grade": "number (2 decimals)",
  "median_grade": "number (2 decimals)",
  "std_dev_grade": "number (2 decimals)",
  "min_grade": "number (2 decimals)",
  "max_grade": "number (2 decimals)",
  "avg_difficulty": "number (3 decimals)",
  "avg_discrimination": "number (3 decimals)",
  "avg_correlation": "number (3 decimals)",
  "pass_rate": "number (3 decimals)",
  "cronbach_alpha": "number (3 decimals)"
}
```

---

## 🤝 Contributing

Have suggestions for the repository feature? Open an issue or PR on GitHub!

**Ideas for future enhancements:**
- Automated trend detection
- Email alerts for anomalous exams
- Built-in comparison reports
- Export to additional formats (PostgreSQL, MongoDB, etc.)
- Dashboard visualizations

---

## 📚 Additional Resources

### Learn More About the Metrics

- [DEFINITIONS.md](definitions.md) - Explanation of all psychometric terms
- [Classical Test Theory on Wikipedia](https://en.wikipedia.org/wiki/Classical_test_theory)
- [Item Analysis Guide](https://testing.byu.edu/handbooks/itemanalysis.pdf)

### External Documentation

- [Airtable API Documentation](https://airtable.com/developers/web/api/introduction)
- [Baserow API Documentation](https://baserow.io/docs/apis%2Frest-api)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

---

## 📞 Support

**Questions or issues?**
- Check the main [README.md](README.md)
- Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md) (if available)
- Open an issue on GitHub with:
  - Your configuration (without sensitive keys!)
  - Error messages
  - Steps to reproduce

---

**Happy tracking!** 📊

Now you can build a comprehensive historical dataset of your exam metrics and make data-driven decisions about assessment quality over time.
