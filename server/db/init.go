package db

import (
	"gorm.io/gorm"
	"gorm.io/driver/mysql"
	"github.com/howardwang15/surveillance/server/models"
)

var DB *gorm.DB

func Init(dsn string) {
	var err error
	DB, err = gorm.Open(mysql.Open(dsn), &gorm.Config{})
	if err != nil {
		panic(err)
	}

	DB.AutoMigrate(&models.TempVideo{})
}
