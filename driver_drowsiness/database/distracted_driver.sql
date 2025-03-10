-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Apr 19, 2022 at 05:27 AM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `distracted_driver`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`) VALUES
('admin', 'admin');

-- --------------------------------------------------------

--
-- Table structure for table `register`
--

CREATE TABLE `register` (
  `id` int(11) NOT NULL,
  `name` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL,
  `address` varchar(50) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `pass` varchar(20) NOT NULL,
  `carno` varchar(20) NOT NULL,
  `rdate` varchar(15) NOT NULL,
  `owner` varchar(20) NOT NULL,
  `utype` varchar(20) NOT NULL,
  `fimg` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `register`
--

INSERT INTO `register` (`id`, `name`, `mobile`, `email`, `address`, `uname`, `pass`, `carno`, `rdate`, `owner`, `utype`, `fimg`) VALUES
(1, 'Ram', 9976570006, 'ram@gmail.com', 'Salem', 'ram', '1234', 'TN2223', '14-04-2022', 'ram', 'owner', '1_40.jpg'),
(2, 'Rahul', 8890145671, 'rahul@gmail.com', 'Madurai', 'Rahul', '', 'TN2223', '14-04-2022', 'ram', 'user', '2_58.jpg');

-- --------------------------------------------------------

--
-- Table structure for table `vt_alert`
--

CREATE TABLE `vt_alert` (
  `id` int(11) NOT NULL,
  `uname` varchar(20) NOT NULL,
  `driver` varchar(20) NOT NULL,
  `message` varchar(50) NOT NULL,
  `dtime` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `vt_alert`
--

INSERT INTO `vt_alert` (`id`, `uname`, `driver`, `message`, `dtime`) VALUES
(1, 'ram', 'Ram', 'Alert: DD - Calling', '2022-04-16 08:29:26'),
(2, 'ram', 'Ram', 'Alert: DD - Calling', '2022-04-16 08:29:37'),
(3, 'ram', 'Ram', 'Alert:DD - Behind', '2022-04-16 08:56:53'),
(4, 'ram', 'Ram', 'Alert:DD - Behind', '2022-04-16 08:57:06'),
(5, 'ram', 'Ram', 'Alert:DD - Behind', '2022-04-16 08:57:19'),
(6, 'ram', 'Ram', 'Alert:DD - Behind', '2022-04-16 08:57:41'),
(7, 'ram', 'Ram', 'Alert:DD - Behind', '2022-04-16 08:57:52'),
(8, 'ram', 'Ram', 'Alert:DD - Behind', '2022-04-16 08:58:04'),
(9, 'ram', 'Ram', 'Alert:DD - Behind', '2022-04-16 08:58:17'),
(10, 'ram', 'Ram', 'Alert: DD - Texting', '2022-04-16 08:58:29'),
(11, 'ram', 'Ram', 'Alert: DD - Texting', '2022-04-16 08:58:40'),
(12, 'ram', 'Ram', 'Alert: DD - Calling', '2022-04-16 09:00:08'),
(13, 'ram', 'Ram', 'Alert: DD - Calling', '2022-04-16 09:00:24');

-- --------------------------------------------------------

--
-- Table structure for table `vt_dd`
--

CREATE TABLE `vt_dd` (
  `id` int(11) NOT NULL,
  `vid` int(11) NOT NULL,
  `vimage` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `vt_dd`
--

INSERT INTO `vt_dd` (`id`, `vid`, `vimage`) VALUES
(120, 1, '1_2.png'),
(121, 1, '1_3.png'),
(122, 1, '1_4.png'),
(123, 1, '1_5.png'),
(124, 1, '1_6.png'),
(125, 1, '1_7.png'),
(126, 1, '1_8.png'),
(127, 1, '1_9.png'),
(128, 1, '1_10.png'),
(129, 1, '1_11.png'),
(130, 1, '1_12.png'),
(131, 1, '1_13.png'),
(132, 1, '1_14.png'),
(133, 1, '1_15.png'),
(134, 1, '1_16.png'),
(135, 1, '1_17.png'),
(136, 1, '1_18.png'),
(137, 1, '1_19.png'),
(138, 1, '1_20.png'),
(139, 1, '1_21.png'),
(140, 1, '1_22.png'),
(141, 1, '1_23.png'),
(142, 1, '1_24.png'),
(143, 1, '1_25.png'),
(144, 1, '1_26.png'),
(145, 1, '1_27.png'),
(146, 1, '1_28.png'),
(147, 1, '1_29.png'),
(148, 1, '1_30.png'),
(149, 1, '1_31.png'),
(150, 1, '1_32.png'),
(151, 1, '1_33.png'),
(152, 1, '1_34.png'),
(153, 1, '1_35.png'),
(154, 1, '1_36.png'),
(155, 1, '1_37.png'),
(156, 1, '1_38.png'),
(157, 1, '1_39.png'),
(158, 2, '2_2.png'),
(159, 2, '2_3.png'),
(160, 2, '2_4.png'),
(161, 2, '2_5.png'),
(162, 2, '2_6.png'),
(163, 2, '2_7.png'),
(164, 2, '2_8.png'),
(165, 2, '2_9.png'),
(166, 2, '2_10.png'),
(167, 2, '2_11.png'),
(168, 2, '2_12.png'),
(169, 2, '2_13.png'),
(170, 2, '2_14.png'),
(171, 2, '2_15.png'),
(172, 2, '2_16.png'),
(173, 2, '2_17.png'),
(174, 2, '2_18.png'),
(175, 2, '2_19.png'),
(176, 2, '2_20.png'),
(177, 2, '2_21.png'),
(178, 2, '2_22.png'),
(179, 2, '2_23.png'),
(180, 2, '2_24.png'),
(181, 2, '2_25.png'),
(182, 2, '2_26.png'),
(183, 2, '2_27.png'),
(184, 2, '2_28.png'),
(185, 2, '2_29.png'),
(186, 2, '2_30.png'),
(187, 2, '2_31.png'),
(188, 2, '2_32.png'),
(189, 2, '2_33.png'),
(190, 2, '2_34.png'),
(191, 2, '2_35.png'),
(192, 2, '2_36.png'),
(193, 2, '2_37.png'),
(194, 2, '2_38.png'),
(195, 2, '2_39.png'),
(196, 2, '2_40.png'),
(197, 3, '3_2.png'),
(198, 3, '3_3.png'),
(199, 3, '3_4.png'),
(200, 3, '3_5.png'),
(201, 3, '3_6.png'),
(202, 3, '3_7.png'),
(203, 3, '3_8.png'),
(204, 3, '3_9.png'),
(205, 3, '3_10.png'),
(206, 3, '3_11.png'),
(207, 3, '3_12.png'),
(208, 3, '3_13.png'),
(209, 3, '3_14.png'),
(210, 3, '3_15.png'),
(211, 3, '3_16.png'),
(212, 3, '3_17.png'),
(213, 3, '3_18.png'),
(214, 3, '3_19.png'),
(215, 3, '3_20.png'),
(216, 3, '3_21.png'),
(217, 3, '3_22.png'),
(218, 3, '3_23.png'),
(219, 3, '3_24.png'),
(220, 3, '3_25.png'),
(221, 3, '3_26.png'),
(222, 3, '3_27.png'),
(223, 3, '3_28.png'),
(224, 3, '3_29.png'),
(225, 3, '3_30.png'),
(226, 3, '3_31.png'),
(227, 3, '3_32.png'),
(228, 3, '3_33.png'),
(229, 3, '3_34.png'),
(230, 3, '3_35.png'),
(231, 3, '3_36.png'),
(232, 3, '3_37.png'),
(233, 3, '3_38.png'),
(234, 3, '3_39.png'),
(235, 4, '4_2.png'),
(236, 4, '4_3.png'),
(237, 4, '4_4.png'),
(238, 4, '4_5.png'),
(239, 4, '4_6.png'),
(240, 4, '4_7.png'),
(241, 4, '4_8.png'),
(242, 4, '4_9.png'),
(243, 4, '4_10.png'),
(244, 4, '4_11.png'),
(245, 4, '4_12.png'),
(246, 4, '4_13.png'),
(247, 4, '4_14.png'),
(248, 4, '4_15.png'),
(249, 4, '4_16.png'),
(250, 4, '4_17.png'),
(251, 4, '4_18.png'),
(252, 4, '4_19.png'),
(253, 4, '4_20.png'),
(254, 4, '4_21.png'),
(255, 4, '4_22.png'),
(256, 4, '4_23.png'),
(257, 4, '4_24.png'),
(258, 4, '4_25.png'),
(259, 4, '4_26.png'),
(260, 4, '4_27.png'),
(261, 4, '4_28.png'),
(262, 4, '4_29.png'),
(263, 4, '4_30.png'),
(264, 4, '4_31.png'),
(265, 4, '4_32.png'),
(266, 4, '4_33.png'),
(267, 4, '4_34.png'),
(268, 4, '4_35.png'),
(269, 4, '4_36.png'),
(270, 4, '4_37.png'),
(271, 4, '4_38.png');

-- --------------------------------------------------------

--
-- Table structure for table `vt_face`
--

CREATE TABLE `vt_face` (
  `id` int(11) NOT NULL,
  `vid` int(11) NOT NULL,
  `vface` varchar(20) NOT NULL,
  `status` int(11) NOT NULL,
  `owner` varchar(20) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `vt_face`
--

INSERT INTO `vt_face` (`id`, `vid`, `vface`, `status`, `owner`) VALUES
(1, 1, '1_2.jpg', 0, 'ram'),
(2, 1, '1_3.jpg', 0, 'ram'),
(3, 1, '1_4.jpg', 0, 'ram'),
(4, 1, '1_5.jpg', 0, 'ram'),
(5, 1, '1_6.jpg', 0, 'ram'),
(6, 1, '1_7.jpg', 0, 'ram'),
(7, 1, '1_8.jpg', 0, 'ram'),
(8, 1, '1_9.jpg', 0, 'ram'),
(9, 1, '1_10.jpg', 0, 'ram'),
(10, 1, '1_11.jpg', 0, 'ram'),
(11, 1, '1_12.jpg', 0, 'ram'),
(12, 1, '1_13.jpg', 0, 'ram'),
(13, 1, '1_14.jpg', 0, 'ram'),
(14, 1, '1_15.jpg', 0, 'ram'),
(15, 1, '1_16.jpg', 0, 'ram'),
(16, 1, '1_17.jpg', 0, 'ram'),
(17, 1, '1_18.jpg', 0, 'ram'),
(18, 1, '1_19.jpg', 0, 'ram'),
(19, 1, '1_20.jpg', 0, 'ram'),
(20, 1, '1_21.jpg', 0, 'ram'),
(21, 1, '1_22.jpg', 0, 'ram'),
(22, 1, '1_23.jpg', 0, 'ram'),
(23, 1, '1_24.jpg', 0, 'ram'),
(24, 1, '1_25.jpg', 0, 'ram'),
(25, 1, '1_26.jpg', 0, 'ram'),
(26, 1, '1_27.jpg', 0, 'ram'),
(27, 1, '1_28.jpg', 0, 'ram'),
(28, 1, '1_29.jpg', 0, 'ram'),
(29, 1, '1_30.jpg', 0, 'ram'),
(30, 1, '1_31.jpg', 0, 'ram'),
(31, 1, '1_32.jpg', 0, 'ram'),
(32, 1, '1_33.jpg', 0, 'ram'),
(33, 1, '1_34.jpg', 0, 'ram'),
(34, 1, '1_35.jpg', 0, 'ram'),
(35, 1, '1_36.jpg', 0, 'ram'),
(36, 1, '1_37.jpg', 0, 'ram'),
(37, 1, '1_38.jpg', 0, 'ram'),
(38, 1, '1_39.jpg', 0, 'ram'),
(39, 1, '1_40.jpg', 0, 'ram');
