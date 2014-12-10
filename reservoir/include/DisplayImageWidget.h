
/**
 * \file DisplayImageWidget.h
 * \brief Defines DisplayImageWidget
 * \author Florian Lance
 * \date 22/01/13
 */


#ifndef _SWDISPLAYIMAGEWIDGET_
#define _SWDISPLAYIMAGEWIDGET_

#include <QPainter>
#include <QWidget>

/**
 * \class DisplayImageWidget
 * \brief Widget displaying a qimage
 * \author Florian Lance
 * \date 22/12/12
 */
class DisplayImageWidget : public QWidget
{
	Q_OBJECT

	public:

		/**
		* \brief default constructor of SWDisplayImageWidget
		* \param [in] oParent : ... 
		*/		
        DisplayImageWidget(QWidget* oParent = 0,  const bool bScaleImage = true, const bool bActiveSelectPixelMode = false);

		/**
		* \brief destructor of SWDisplayImageWidget
		*/	    
        ~DisplayImageWidget();


        void resizeEvent ( QResizeEvent * event );

	public slots:

		/**
		* \brief Set the new image and update the display.
		* \param [in] oQImage : qimage to display in the widget
		*/		
		void refreshDisplay(const QImage &oQImage);

        /**
         * @brief resetSelectedPoints
         */
        void resetSelectedPoints();


	protected:

		/**
		* \brief paint event
		*/		    
		void paintEvent(QPaintEvent *);

        /**
         * @brief mousePressEvent
         * @param event
         */
        void mousePressEvent ( QMouseEvent * event );

        /**
         * @brief mouseReleaseEvent
         * @param event
         */
        void mouseReleaseEvent ( QMouseEvent * event );

        /**
         * @brief mouseMoveEvent
         * @param event
         */
        void mouseMoveEvent(QMouseEvent * event);

    signals :

        void clickPoint(QPoint, QSize, bool);


	private:

		QImage m_oQImage;	/**< rgb image to display */
        QImage m_oScaledImage;

        QSize m_oSize;  /**< ... */

        bool m_bMouseClicked;
        bool m_bMouseLeftClick;
        bool m_bMouseRightClick;

        bool m_bScaleImage;

        bool m_bActiveSelectPixelMode;

    public :
        QVector<QPoint> m_vClickedPoints;
        QVector<QSize> m_vCurrentSize;
};


#endif 