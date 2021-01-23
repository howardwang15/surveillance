package operations

import (
	"github.com/howardwang15/surveillance/server/db"
	"github.com/howardwang15/surveillance/server/models"
	"html/template"
	"net/http"
	"path"
	"strconv"
	"time"
)

func formatTime(ts time.Time) string {
	layout := "1/2/2006 3:04:05 PM"
	return ts.Format(layout)
}

type VideoResponse struct {
	Time      string
	ImageFile string
	VideoFile string
	ID        uint64
}

func RenderHomePage(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query()
	page := query["page"]
	pageFetch := 1

	if len(page) != 0 {
		var err error
		pageFetch, err = strconv.Atoi(page[0])
		if err != nil {
			pageFetch = 1
		}
	}

	var videos []models.TempVideo
	NUM_RESULTS := 20
	db.DB.Order("Id desc").Limit(NUM_RESULTS).Offset((pageFetch - 1) * NUM_RESULTS).Find(&videos)

	videoResponses := make([]VideoResponse, len(videos))
	for i := 0; i < len(videos); i++ {
		videoResponses[i] = VideoResponse{formatTime(videos[i].StartTime),
			path.Join("assets", videos[i].FirstFrame),
			path.Join("assets", videos[i].VideoName),
			videos[i].Id,
		}
	}

	template, _ := template.New("photo_gallery.html").Funcs(template.FuncMap{
		"htmlSafe": func(html string) template.HTML {
			return template.HTML(html)
		},
	}).ParseFiles("templates/photo_gallery.html")
	template.Execute(w, videoResponses)
}
