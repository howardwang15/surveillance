package models

import (
	"gorm.io/gorm"
	"time"
)

type TempVideo struct {
	gorm.Model
	Id uint64 `gorm:"primaryKey;autoIncrement"`
	StartTime time.Time
	VideoName string
	FirstFrame string
}